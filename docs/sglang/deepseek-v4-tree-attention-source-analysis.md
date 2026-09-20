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
| DeepSeek-V4 `topk=1` guard 限在哪一层 | 它是**端到端 capability gate**。模型配置层先全局禁止 DeepSeek-V4 EAGLE topk>1；同时 CUD