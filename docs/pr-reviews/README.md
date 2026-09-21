# SGLang PR 精读

从真实 merged PR 学 SGLang 的设计、Bug、性能优化与工程取舍。

这个栏目不做“更新日志搬运”，而是优先选择改动范围清晰、因果链完整、能够引出一个核心知识点的 PR。每篇都区分：

- PR 实际修改了什么；
- 为什么原实现会出问题；
- 对应的源码 invariant / ownership / topology 是什么；
- 哪些结论只适用于特定模型、设备或配置；
- 能迁移到其他 SGLang / AI Infra 场景的工程经验。

## 已发布

- [PR #39871：为什么 TP16 不是 AttnTP16？](sglang-pr-39871-attention-parallel-widths.md)：从一个日志修复切入，理解 DP Attention 下的 `tp_size / attn_dp_size / attn_cp_size / attn_tp_size`，以及 Config Value、Runtime Derived Value 与 Single Source of Truth。
- [PR #39120：一个 Tensor 明明很小，为什么却偷偷占着整个 Batch？](sglang-pr-39120-tensor-view-storage-cache.md)：从 `torch.split` View 保活整块 Batch Storage 的问题切入，理解 Logical Payload、Backing Storage、Ownership、Lifetime 与 Cache Memory Accounting。

## 怎么读这个栏目

如果你刚开始读 SGLang 源码，建议优先看改动小、边界清楚的 PR：先建立“现象 → invariant → 修复 → 可迁移知识”的阅读方法，再进入 Scheduler、KV Cache、Speculative Decoding、PD 和 Kernel 优化等更大的改动。

[返回仓库首页](../../README.md)
