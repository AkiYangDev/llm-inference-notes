# DeepSeek-V4 Speculative Decoding 源码解析：Verify 为什么需要 Tree Attention？Tree Mask、Position 与 Candidate Layout 如何协同

上一篇我们已经把 Speculative Decoding 的 KV 生命周期追到了：

~~~text
Draft
  ↓
Target Verify
  ↓
Accept
  ↓
KV Commit / Compact
  ↓
下一轮
~~~

但这里还有一个更基础的问题没有真正展开：

> **Target Model 一次 Verify 多个 speculative candidate 时，每个 candidate 到底应该看见哪些历史 KV？**

如果 Draft 只有一条链：

~~~text
root
 ↓
 A
 ↓
 B
 ↓
 C
~~~

普通 causal attention 就够了。

但 EAGLE 一旦进入真正的多分支候选：

~~~text
               root
             /      \
            A        B
            |        |
            C        D
~~~

如果只是把：

~~~text
[root, A, B, C, D]
~~~

按物理 row 顺序塞进普通 causal attention，B 会错误看到 A，C 会错误看到 B，D 更会看到另一条 branch 上的 A/C。

此时问题已经不是性能，而是 correctness：

~~~text
Target logits
不再对应
某一条真实 root-to-node path
~~~

于是后面的 Accept 即使 tree traversal 完全正确，也是在验证一批已经被 sibling branch 污染的 logits。

Tree Attention 真正解决的，就是：

> **怎样在一次 Target Forward 里，同时计算多条互斥的未来，又让每个 candidate 只生活在属于自己的那条 causal history 中。**

本文沿下面这条源码链展开：

~~~text
Draft Expansion
      ↓
Candidate Selection
      ↓
Compact Verify Rows
      ↓
Parent Topology
      ↓
Tree Position
      ↓
Attention Visibility
      ↓
Target Logits
      ↓
Tree Traversal / Accept
~~~

> **源码基线**：本文基于 `sgl-project/sglang @ 791c7850d0960fd768102f71e7d999b036bb75ba`，核对日期为 2026-09-20。重点覆盖 EAGLE Spec V2 的 tree builder、VerifyMask、Triton / FlashAttention target-verify，以及 DeepSeek-V4 DSV4 的当前 capability gate。
>
> **DeepSeek-V4 当前边界**：固定版本中，DeepSeek-V4 的 EAGLE 配置仍要求 `speculative_eagle_topk == 1`；CUDA/HIP DSV4 Attention backend 还会独立 assert `topk in [0, 1]`。因此本文的 multi-branch Tree Attention 是对 **SGLang 通用 EAGLE Tree Verify 机制，以及 DeepSeek-V4 若未来放开 topk>1 所必须满足的 correctness contract** 的源码解析，而不是声称当前 DeepSeek-V4 生产路径已经在跑多分支 Tree Attention。

## Strict Review 结论先行

这次发布前重点复核了四个容易写错的点。

| Review 点 | 固定源码结论 |
| --- | --- |
| `FULL_MASK / QLEN_ONLY` 的 prefix 语义 | 对**支持 EAGLE tree verify 的 backend**，逻辑语义是一致的：candidate 可看完整 KV-ready prefix，再加自己的 ancestor path。`FULL_MASK / QLEN_ONLY` 是不同物化方式，不是跨 backend 的统一 kernel ABI；而且有的 backend 根本不读取 mask buffer。 |
| root / bonus 的 Position 与 KV-ready boundary | 上一轮 terminal bonus 在下一轮成为 root。它虽然已经作为 token 输出，但其 KV 尚未进入 `[0, seq_len)`；下一轮 tree builder 将它放在 `position = seq_len`，Target Verify 才真正为它写 KV。当前轮新 terminal bonus 再次留在新的 KV-ready boundary 外。 |
| FlashAttention cascade 能否称为 Tree Attention | **可以称为 Tree Attention 的一种 factorized implementation**：common prefix attention + per-query ancestor suffix attention + LSE-aware merge。但它不是“一个 Tree Attention kernel”；SWA 层还明确不走这条 cascade。 |
| DeepSeek-V4 `topk=1` guard 限在哪一层 | 它是**端到端 capability gate**。模型配置层先全局禁止 DeepSeek-V4 EAGLE topk>1；同时 CUDA/HIP `DeepseekV4AttnBackend` 自身也明确不支持 topk>1。因此不能只归因于配置 guard，也不能只归因于 Attention；Attention 是已确认的阻塞点之一，完整 DSV4 compressed-KV / state pipeline 仍需整体证明。 |

下面逐层展开。

---

## 一、Tree Attention 的本质：一次 Forward 中模拟多条互斥的 Causal History

先看最简单的例子。

假设 prefix 已经确定：

~~~text
P0 P1 P2 ... Pn
~~~

Draft Model 提出：

~~~text
                 root
               /      \
              A        B
              |        |
              C        D
~~~

如果不用 Tree Attention，最直接但最慢的做法是：

~~~text
Target(prefix + root + A + C)

Target(prefix + root + B + D)
~~~

两条 branch 分别跑一次 Target Model。

这样一定正确，但 speculative decoding 想减少的 Target Forward 次数又回来了。

我们真正想要的是：

~~~text
一次 Target Forward

物理输入：
[root, A, B, C, D]

逻辑语义：
root → A → C

以及

root → B → D
~~~

也就是说：

> **物理上把候选打包在一个 batch / token block 中，逻辑上仍然把它们视为多条不同 sequence。**

### 普通 causal mask 为什么会错

如果 Verify row 顺序是：

~~~text
row 0 = root
row 1 = A
row 2 = B
row 3 = C
row 4 = D
~~~

普通 lower-triangular causal mask 会得到：

~~~text
          K
          0 1 2 3 4

Q 0       1 0 0 0 0
  1       1 1 0 0 0
  2       1 1 1 0 0
  3       1 1 1 1 0
  4       1 1 1 1 1
~~~

于是：

~~~text
B 看见 A
C 看见 B
D 看见 A/B/C
~~~

这些都是 sibling / cross-branch leakage。

正确的 candidate visibility 应该是：

~~~text
          K
          0 1 2 3 4

Q 0       1 0 0 0 0
  1       1 1 0 0 0
  2       1 0 1 0 0
  3       1 1 0 1 0
  4       1 0 1 0 1
~~~

Tree Attention 的核心规则可以写成：

M(i,j)=1

当且仅当：

~~~text
j 属于 committed prefix

或者

j == i

或者

j 是 i 的 ancestor
~~~

否则：

M(i,j)=0

因此 Tree Attention 与普通 causal attention 的根本区别是：

~~~text
普通 causal：
“物理上排在我前面”
        ↓
“属于我的历史”

Tree Attention：
“逻辑上是我的 ancestor”
        ↓
“属于我的历史”
~~~

### 为什么这直接决定 Accept correctness

Target Verify 最终希望满足：

~~~text
L(v)
=
Target(prefix + path(root → v))
~~~

也就是说，node `v` 对应的 logits 必须来自它自己的 root-to-v history。

如果 Attention 阶段已经让 sibling 泄漏进来：

~~~text
L(v)
=
Target(prefix + physical rows before v)
~~~

那么后面的：

~~~text
retrieve_next_token
retrieve_next_sibling
verify_tree_greedy
~~~

再正确都没有意义。

因为 traversal 在一棵树上走，而 logits 是在另一套 history 上算出来的。

所以：

> **Tree Attention 是 Tree Accept 成立之前的 correctness 前提。**

---

## 二、Candidate Layout：Draft Expansion Pool 怎样被压成 Target Verify Row

Tree Mask 不是凭空生成的。

它必须先知道：

~~~text
每一个 Verify row
到底是哪一个 candidate node
~~~

当前 EAGLE V2 的 Draft 主入口在：

[`EagleWorkerV2.draft_forward()`](https://github.com/sgl-project/sglang/blob/791c7850d0960fd768102f71e7d999b036bb75ba/python/sglang/srt/speculative/eagle_worker_v2.py)

多步 Draft 会不断维护：

~~~text
score_list
token_list
parents_list
~~~

后续 candidate score 不是单步概率，而是累计 path score。

在：

[`_select_top_k_tokens_later()`](https://github.com/sgl-project/sglang/blob/791c7850d0960fd768102f71e7d999b036bb75ba/python/sglang/srt/speculative/spec_utils.py)

中：

~~~python
expand_scores =
    scores.unsqueeze(2)
    * topk_p.view(-1,