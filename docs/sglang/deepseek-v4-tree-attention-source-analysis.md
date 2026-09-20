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
    * topk_p.view(-1, topk, topk)
~~~

例如：

~~~text
root
├─ A 0.60
│  ├─ C 0.70  → path score 0.42
│  └─ E 0.20  → path score 0.12
│
└─ B 0.40
   ├─ D 0.80  → path score 0.32
   └─ F 0.10  → path score 0.04
~~~

最终并不是把整个 expansion tree 都交给 Target。

`organize_draft_results()` 会：

~~~text
1. flatten score pool
2. top-k 选出 num_draft_token - 1 个 candidate
3. sort selected indices
4. gather 真正进入 Verify 的 draft tokens
5. 保留 parent topology
~~~

固定源码：

[`organize_draft_results()`](https://github.com/sgl-project/sglang/blob/791c7850d0960fd768102f71e7d999b036bb75ba/python/sglang/srt/speculative/eagle_utils.py)

这里最需要区分两个 index space：

~~~text
top_scores_index / selected_index
        │
        └─ Draft expansion candidate-pool space

retrieve_index
        │
        └─ Compact Target-Verify row space
~~~

可以画成：

~~~text
Draft Expansion Pool

#0 #1 #2 #3 #4 #5 #6 ...
        │
        │ top_scores_index
        ▼
Selected Candidate Set

#0 #1 #3 #5 ...
        │
        │ build_tree_kernel
        ▼
Compact Verify Rows

row0 row1 row2 row3 ...
~~~

Tree Attention 工作的对象，是最后这套：

~~~text
Compact Verify Row
~~~

### Verify root 不是普通 draft candidate

进入：

[`build_tree_kernel_efficient()`](https://github.com/sgl-project/sglang/blob/791c7850d0960fd768102f71e7d999b036bb75ba/python/sglang/srt/speculative/eagle_utils.py)

后第一步就是：

~~~python
draft_tokens = torch.cat(
    (
        bonus_tokens.unsqueeze(1),
        draft_tokens,
    ),
    dim=1,
).flatten()
~~~

所以 Target Verify 实际输入是：

~~~text
[上一轮 bonus/root, selected draft candidates...]
~~~

而不是单纯：

~~~text
[draft1, draft2, draft3...]
~~~

这个 root/bonus 的跨轮语义，后面单独展开。

---

## 三、Tree Position：Sibling 为什么必须拥有相同的时间坐标

Tree Visibility 只解决：

~~~text
“我可以看谁？”
~~~

但模型还需要知道：

~~~text
“我处在第几个 token position？”
~~~

考虑：

~~~text
root
├─ A
│  └─ C
└─ B
   └─ D
~~~

假设当前 KV-ready prefix 长度是：

~~~text
P
~~~

正确 Position 必须是：

~~~text
root → P

A    → P + 1
B    → P + 1

C    → P + 2
D    → P + 2
~~~

而不能按 physical row 编成：

~~~text
root → P
A    → P+1
B    → P+2
C    → P+3
D    → P+4
~~~

因为：

~~~text
A / B
~~~

不是“先发生 A，再发生 B”。

它们是：

> **同一个未来时间步的两个备选世界。**

因此：

~~~text
same tree depth
=
same logical position
~~~

### 源码确实按 depth 算 Position

Tree kernel 在 root 位置写：

~~~python
positions[root] = seq_len
~~~

其它 node 则沿 parent relation 向上回溯，统计 depth：

~~~text
position
=
seq_len + tree_depth
~~~

固定 Triton 实现：

[`spec_tree.py`](https://github.com/sgl-project/sglang/blob/791c7850d0960fd768102f71e7d999b036bb75ba/python/sglang/kernels/ops/speculative/spec_tree.py)

源码注释给出的例子就是：

~~~text
depth:
[0, 1, 1, 2]

prefix length:
7

positions:
[7, 8, 8, 9]
~~~

SGLang 的真实单测更进一步。

[`test_build_eagle_tree.py`](https://github.com/sgl-project/sglang/blob/791c7850d0960fd768102f71e7d999b036bb75ba/test/registered/spec/utils/test_build_eagle_tree.py)

其中一个 request 得到：

~~~text
positions =
[5, 6, 6, 7, 7, 8, 8, 9]
~~~

配合：

~~~text
retrieve_next_token =
[1, 3, 4, 5, 6, 7, -1, -1]

retrieve_next_sibling =
[-1, 2, -1, -1, -1, -1, -1, -1]
~~~

对应的树是：

~~~text
                   row0 pos5
                  /                   row1 pos6          row2 pos6
              |                  |
          row3 pos7          row4 pos7
              |                  |
          row5 pos8          row6 pos8
              |
          row7 pos9
~~~

### Mask 正确而 Position 错，仍然会错

假设 B 的 mask 已经保证它只看：

~~~text
prefix + root + B
~~~

但误把 B 的 position 编成：

~~~text
P + 2
~~~

而不是：

~~~text
P + 1
~~~

对于 RoPE 模型，Q/K rotation 已经变了。

于是：

~~~text
Position 错
  ↓
RoPE phase 错
  ↓
QK score 错
  ↓
hidden state 错
  ↓
Target logits 错
  ↓
Accept 错
~~~

所以：

~~~text
Tree Mask
回答：
“我能看谁？”

Tree Position
回答：
“我处在哪个未来时间步？”
~~~

两者缺一不可。

---

## 四、Root / Bonus 与 KV-ready Boundary：跨轮时其实存在一枚 Token 的 Frontier Gap

这是这次严格 Review 后，正文最需要强化的一点。

直觉上容易认为：

~~~text
上一轮 bonus 已经输出
        ↓
它一定已经在 KV Cache 里
~~~

但 EAGLE 的流水线不是这样。

### `batch.seq_lens` 在这里更接近 KV-ready boundary

`prepare_for_draft()` 对 paged tree layout 有一段非常直接的源码注释：

~~~text
base is batch.seq_lens
(== KV-ready committed prefix at draft time;
the bonus is the tree root written by verify,
not part of [0:seq_lens])
~~~

固定源码：

[`prepare_for_draft()`](https://github.com/sgl-project/sglang/blob/791c7850d0960fd768102f71e7d999b036bb75ba/python/sglang/srt/speculative/eagle_worker_common.py)

这句话非常关键。

假设当前：

~~~text
batch.seq_lens = P
~~~

表示：

~~~text
positions [0, P)
~~~

已经是 KV-ready prefix。

上一轮产生的 terminal bonus：

~~~text
R
~~~

虽然已经是下一枚逻辑 token，但它的 KV 还没有进入这个 prefix。

所以下一轮：

~~~text
R
~~~

会成为 Verify Tree 的 root：

~~~text
position(R) = P
~~~

同时 `eagle_prepare_for_verify()` 从：

~~~python
start_offset=batch.seq_lens
~~~

开始给 Verify rows 分配写入位置。

固定源码：

[`eagle_prepare_for_verify()`](https://github.com/sgl-project/sglang/blob/791c7850d0960fd768102f71e7d999b036bb75ba/python/sglang/srt/speculative/eagle_utils.py)

也就是说：

~~~text
上一轮 terminal bonus
        │
        │ 已输出 / 已成为逻辑 frontier
        ▼
下一轮 root
        │
        │ position = old seq_len
        ▼
本轮 Target Verify
        │
        ▼
此时才真正生成 root KV
~~~

### 为什么 `accept_lens` 同时能推进 KV-ready boundary

用一个完整例子。

当前 KV-ready prefix：

~~~text
[0, P)
~~~

上一轮 terminal bonus：

~~~text
R
~~~

本轮 Draft 在 R 后提出：

~~~text
D1
D2
D3
~~~

Target Verify 的**输入 rows**是：

~~~text
row0 = R
row1 = D1
row2 = D2
row3 = D3
~~~

对应 Position：

~~~text
P
P+1
P+2
P+3
~~~

而每个 row 的 logits 分别预测：

~~~text
row0(R)  → predicts D1
row1(D1) → predicts D2
row2(D2) → predicts D3
row3(D3) → predicts new bonus B
~~~

如果 D1/D2/D3 都被接受，那么：

~~~text
num_correct_drafts = 3
accept_lens = 4
~~~

用户侧本轮 accepted outputs 是：

~~~text
D1 D2 D3 B
~~~

但是本轮真正算出、可以进入 KV-ready prefix 的 Verify input rows 是：

~~~text
R D1 D2 D3
~~~

同样正好 4 个。

所以：

~~~python
new_seq_lens =
    batch.seq_lens + accept_lens
~~~

数值上完全正确：

~~~text
new KV-ready boundary
=
P + 4
~~~

它已经覆盖：

~~~text
R D1 D2 D3
~~~

但**没有**覆盖新 terminal bonus：

~~~text
B
~~~

B 再次停在新的 KV-ready boundary 上：

~~~text
position(B) = P + 4
~~~

等待下一轮作为 root 被真正 Forward。

因此每轮都存在这样一个 frontier：

~~~text
KV-ready prefix
|=======================|

                        B
                        ▲
                  已输出的 terminal bonus
                  但 KV 尚未 materialize
~~~

下一轮：

~~~text
|=======================| B
                          │
                          ▼
                    Verify root
                          │
                          ▼
                    materialize KV
~~~

这也是为什么“bonus”与“root”不是两个不同 token。

更准确地说：

> **当前轮 terminal bonus，就是下一轮 Tree root。**

### 这也解释了上一篇中的一个看似奇怪现象

上一篇我们看到：

~~~text
accept_lens
=
num_correct_drafts + 1
~~~

而 accepted KV relocation 同样处理 `accept_lens` 个 Verify rows。

现在可以更准确地理解这两个同样的数字。

它们对应的是两组**错开一格**的 token：

~~~text
本轮 accepted outputs：
D1 D2 D3 B

本轮 committed Verify/KV rows：
R  D1 D2 D3
~~~

两边长度都等于：

~~~text
4
~~~

但 token identity 向前错了一格。

这个 one-token shift 是整个 EAGLE pipeline 最值得记住的 invariant 之一。

---

## 五、FULL_MASK 与 QLEN_ONLY：语义一致，不代表所有 Backend 都读同一种 Mask

现在终于可以精确回答第一个 Review 问题。

当前 SGLang 定义：

~~~python
class TreeMaskMode(IntEnum):
    FULL_MASK = 0
    QLEN_ONLY = 1
    QLEN_ONLY_BITPACKING = 2
~~~

固定源码：

[`eagle_utils.py`](https://github.com/sgl-project/sglang/blob/791c7850d0960fd768102f71e7d999b036bb75ba/python/sglang/srt/speculative/eagle_utils.py)

### FULL_MASK：显式包含 Prefix + Tree Block

对于一个 request，FULL_MASK 的逻辑形状近似：

~~~text
[num_verify_tokens,
 seq_len + num_verify_tokens]
~~~

例如：

~~~text
                 Prefix KV                  Candidate KV
        ┌────────────────────────┬─────────────────────────┐
        │ P0 P1 ... P{L-1}      │ 0 1 2 3 4 ...          │
┌───────┼────────────────────────┼─────────────────────────┤
│ Q0    │ 1  1 ... 1            │ tree ancestry           │
│ Q1    │ 1  1 ... 1            │ tree ancestry           │
│ Q2    │ 1  1 ... 1            │ tree ancestry           │
└───────┴────────────────────────┴─────────────────────────┘
~~~

也就是说：

~~~text
Prefix:
所有 candidate 都可见

Candidate block:
只允许 self + ancestors
~~~

FULL_MASK 的 size 大致是：

~~~text
B × N × (L + N)
~~~

所以长上下文下很贵。

`verify_mask.py` 直接写道：

~~~text
FULL_MASK reaches 100s of MB at long context
~~~

### QLEN_ONLY：只物化 Candidate × Candidate Tree Block

QLEN_ONLY 只保留：

~~~text
[N, N]
~~~

例如：

~~~text
        0 1 2 3 4

0       1 0 0 0 0
1       1 1 0 0 0
2       1 0 1 0 0
3       1 1 0 1 0
4       1 0 1 0 1
~~~

Prefix 不在这块 Tensor 中显式展开。

因此它把 mask storage 从：

~~~text
O(B × N × (L + N))
~~~

降成：

~~~text
O(B × N²)
~~~

### 但严格 Review 后，不能写成“所有 backend 都在 FULL 与 QLEN 两者中二选一读取”

当前还有：

[`VerifyMask`](https://github.com/sgl-project/sglang/blob/791c7850d0960fd768102f71e7d999b036bb75ba/python/sglang/srt/layers/attention/verify_mask.py)

它除了：

~~~text
buffer
mode
max_bs
~~~

还带：

~~~python
is_read
~~~

这代表一个很重要的工程事实：

> **tree builder 可能需要一个 buffer 来统一产生 tree-layout 副产物，但目标 Attention backend 不一定真的读取这块 mask。**

例如当前 CUDA `DeepseekV4AttnBackend`：

~~~text
Verify metadata never extracts the mask.
~~~

然后：

~~~python
maybe_create_verify_mask(
    ...,
    is_read=False,
)
~~~

固定源码：

[`DeepseekV4AttnBackend.init_cuda_graph_state()`](https://github.com/sgl-project/sglang/blob/791c7850d0960fd768102f71e7d999b036bb75ba/python/sglang/srt/layers/attention/deepseek_v4_backend.py)

因为 `is_read=False`，`maybe_create_verify_mask()` 会选择更小的：

~~~text
QLEN_ONLY
~~~

但这里不能反过来说：

~~~text
“DSV4 正在用 QLEN_ONLY mask 做 multi-branch Tree Attention”
~~~

当前 DSV4 本来就禁止 `topk>1`，而且这块 mask 根本不被 Verify metadata 提取。

因此最严谨的结论是：

> **对于支持 EAGLE tree verify 的 backend，逻辑 contract 是一致的：完整 KV-ready prefix 可见、candidate 部分只见 ancestor path；但 FULL_MASK / QLEN_ONLY / page-table rewrite / retrieve topology 都只是不同实现 representation。不是所有 backend 都支持 Tree，也不是所有支持路径都直接读取同一种 mask。**

### Triton 是最直观的 Direct-Mask 实现

Triton Target Verify 直接：

~~~python
custom_mask = spec_info.custom_mask
~~~

随后把：

~~~text
custom_mask
mask_indptr
qo_indptr
kv_indptr
kv_indices
~~~

交给 `extend_attention_fwd()`。

固定源码：

[`triton_backend.py`](https://github.com/sgl-project/sglang/blob/791c7850d0960fd768102f71e7d999b036bb75ba/python/sglang/srt/layers/attention/triton_backend.py)

它最接近：

~~~text
Tree
 ↓
FULL_MASK
 ↓
Attention Kernel
~~~

### 线性 / Hybrid backend 甚至可能传 Tree Topology 而不是 Bool Mask

例如 Ascend hybrid linear attention 对 `topk > 1` 的 target verify 会携带：

~~~text
retrieve_next_token
retrieve_next_sibling
retrieve_parent_token
~~~

说明某些 state-space / linear 路径更适合直接消费 topology metadata。

因此：

~~~text
Tree Attention semantics
≠
必须存在一张 bool Tree Mask
~~~

---

## 六、FlashAttention Cascade：可以叫 Tree Attention，但必须说清它是怎样实现的

这是第三个严格 Review 点。

当前 FlashAttention backend 在：

~~~python
forward_batch.forward_mode.is_target_verify()
and self.topk > 1
and not is_swa_layer
~~~

时设置：

~~~python
use_cascade_attn = True
~~~

源码注释直接写：

~~~text
We do cascade attention for Target Verify with topk > 1
~~~

固定源码：

[`flashattention_backend.py`](https://github.com/sgl-project/sglang/blob/791c7850d0960fd768102f71e7d999b036bb75ba/python/sglang/srt/layers/attention/flashattention_backend.py)

但这里并没有一个名叫：

~~~text
tree_attention(...)
~~~

的单一 kernel。

它实际上把 Tree Attention 分成两部分。

### 第一部分：所有 Query 共享的 Common Prefix

对于 topk>1，第一份 metadata：

~~~text
target_verify_metadata_topk_normal
~~~

只让 query attend：

~~~text
committed prefix
~~~

这里所有 tree node 的 prefix context 都完全相同，所以不需要为每个 branch 重复存整份 mask。

### 第二部分：每个 Query 自己的 Ancestor Suffix

源码从 `custom_mask` 中抽出 candidate block：

~~~python
mask =
    spec_info.custom_mask[
        mask_extraction_indices
    ].view(
        -1,
        speculative_num_draft_tokens,
    )
~~~

然后把：

~~~text
mask == True
~~~

的 candidate physical slots 排到每个 query 自己的 suffix page table 前面。

源码中的注释例子非常直观。

原始 candidate slots：

~~~text
[8, 9, 10]
~~~

Tree mask：

~~~text
query0 [1, 0, 0]
query1 [1, 1, 0]
query2 [1, 0, 1]
~~~

最后可以变成：

~~~text
query0:
[8]
len = 1

query1:
[8, 9]
len = 2

query2:
[8, 10]
len = 2
~~~

于是普通 Attention kernel 不需要理解“Tree”。

它只看到：

> **每个 query 有一份已经筛好的合法 suffix KV list。**

所以这里的工程转换是：

~~~text
Tree Topology
      ↓
Tree Mask
      ↓
Per-query Ancestor KV List
      ↓
普通 FlashAttention
~~~

### Prefix 与 Suffix 不能直接相加

第一段 Attention 得到：

~~~text
(O_prefix, LSE_prefix)
~~~

第二段得到：

~~~text
(O_tree, LSE_tree)
~~~

最终用：

~~~python
merge_state_v2_wrapper(...)
~~~

合并。

因为两个 KV subset 的 softmax normalization 必须在同一个 partition function 下恢复。

所以不能写成：

~~~text
O = O_prefix + O_tree
~~~

而必须结合 LSE。

最终效果等价于：

~~~text
Attention(
    query,
    committed prefix
    ∪
    allowed ancestor suffix
)
~~~

因此：

> **把这条路径称为 Tree Attention 是准确的，只要明确它指的是语义；更精确的工程描述是“FlashAttention 用 prefix/suffix cascade + softmax-state merge 实现 Tree Attention”。**

不要写成：

> “FlashAttention 调用了一个 Tree Attention kernel。”

那就不准确了。

### SWA 是一个重要例外

源码还明确：

~~~text
We don't use cascade attention for Sliding Window Attention
~~~

原因包括：

~~~text
不同 query 需要不同 window size，
而 FA3 cascade interface 不能传一组不同 window sizes。
~~~

所以 SWA 层不走这条 common-prefix cascade，而使用展开后的 spec metadata。

因此文章应该说：

~~~text
FlashAttention 非-SWA topk>1 Target Verify
→ cascade Tree Attention

FlashAttention SWA topk>1
→ 另一套 expanded metadata / page-table 路径
~~~

而不能把整个 FlashAttention backend 全部概括成 cascade。

---

## 七、DeepSeek-V4 为什么仍然卡在 `topk=1`：不是一个 Assert，也不是单一 Attention 问题

最后回到这篇标题里的 DeepSeek-V4。

当前模型级 hook：

[`deepseek_v4_hook.py`](https://github.com/sgl-project/sglang/blob/791c7850d0960fd768102f71e7d999b036bb75ba/python/sglang/srt/arg_groups/deepseek_v4_hook.py)

明确：

~~~python
if cfg.speculative_algorithm == "EAGLE":
    assert cfg.speculative_eagle_topk == 1
~~~

表面看起来：

~~~text
删掉 assert
~~~

似乎就能尝试 `topk>1`。

严格 Review 后，这个理解是不够的。

### 第一层：模型配置层已经把能力整体封死

这个 guard 在 runtime 真正进入 EAGLE Tree Verify 前就拒绝。

它表达的是：

> **当前 Deepseek-V4 EAGLE 对外暴露的 capability contract 是 chain-only。**

它不是某个 kernel 的局部 fallback。

### 第二层：CUDA/HIP DSV4 Attention 本身也明确拒绝 Tree

CUDA：

[`DeepseekV4AttnBackend`](https://github.com/sgl-project/sglang/blob/791c7850d0960fd768102f71e7d999b036bb75ba/python/sglang/srt/layers/attention/deepseek_v4_backend.py)

初始化直接：

~~~python
self.topk = get_spec().speculative_eagle_topk or 0

assert self.topk in [0, 1], (
    "MTP Topk > 1 not supported for DeepSeek V4"
)
~~~

HIP radix backend 也有同样限制。

因此至少可以确认：

> **DSV4 Attention backend 是当前 DeepSeek-V4 multi-branch EAGLE 的一个真实 blocking layer。**

不是只有上层 guard 没放开而已。

### 但也不能反过来说“把 Attention 支持补上就全部解决”

DeepSeek-V4 的 Target Attention 不只是普通 full KV。

还涉及：

~~~text
SWA
C4
C128
DSA / Indexer
Compressor state
paged KV ownership
speculative KV commit
~~~

如果未来真正允许：

~~~text
topk >