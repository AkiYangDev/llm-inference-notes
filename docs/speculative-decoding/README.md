# 投机解码

Speculative Decoding 的 Draft / Verify、接受策略、Verify 调度、缓存提交与状态一致性。这里重点关注 DeepSeek-V4 与 SGLang 中 speculative state 如何进入 Runtime。

## 核心执行链

- [DeepSeek-V4 DSpark：一轮 Draft、Verify、Accept 与 Commit](deepseek-v4-speculative-decoding-source-analysis.md)：从 DSpark Proposal 到 Target Verify、Accept / Reject、commit 与下一轮 Draft state。
- [SGLang 投机解码调度：请求进度、设备状态与跨轮交接](sglang-speculative-decoding-scheduler-source-analysis.md)：解释 overlap 多时钟、publish fence、forward isolation 与 mixed-tail late binding。
- [DeepSeek-V4 MTP：NextN 怎样逐步生成候选](deepseek-v4-mtp-speculative-decoding-source-analysis.md)：从 bundled MTP 权重、Draft Extend 到 multi-step NextN chain。
- [EAGLE 候选树：从候选池到 Verify 与接受路径](deepseek-v4-eagle-draft-tree-source-analysis.md)：从 candidate pool、parent metadata 与路径 score 恢复 Verify Tree。
- [DeepSeek-V4 Speculative Decoding 源码解析：Verify 为什么需要 Tree Attention？](deepseek-v4-tree-attention-source-analysis.md)：解释 Tree Mask、Position 与 ancestor-only visibility。
- [DeepSeek-V4 Speculative Decoding 源码解析：Accept Decision 如何选择最终路径？](deepseek-v4-accept-decision-source-analysis.md)：区分 Greedy Tree Match、Target-only Tree Sampling 与经典 rejection sampling。
- [DeepSeek-V4 Speculative Decoding 源码解析：Accept 后缓存如何 Commit、Compact 与 Rollback？](deepseek-v4-speculative-kv-commit-compact-rollback-source-analysis.md)：把 accepted path 落到 committed KV 与缓存生命周期。
- [DSpark 自适应 Verify：怎样分配候选预算](deepseek-v4-dspark-adaptive-verify-scheduler.md)：从 Confidence / STS 到 SPS Budget、Ragged Verify 与 Ascend 910C compact 边界。

## 跨专题延伸

- [DSpark Parallel Layout](../distributed/deepseek-v4-dspark-parallel-layout-analysis.md)：Draft / Target / Verify 的 TP / DP / EP 布局。
- [投机解码多卡一致性](../distributed/deepseek-v4-speculative-distributed-consistency-source-analysis.md)：SpecTpSync、Accept broadcast 与多卡 control state。
- [Speculative Decoding 性能模型](../performance/deepseek-v4-speculative-performance-analysis.md)：Acceptance Rate、Draft Cost、Verify Cost 与 break-even。
- [Ascend 910C Critical Path](../performance/deepseek-v4-speculative-ascend-performance-critical-path-analysis.md)：NPU Graph、Multi-Stream 与 Communication Overlap。
- [Ascend 910C Profiling 指南](../performance/deepseek-v4-speculative-decoding-ascend-profiling.md)：把 Draft / Verify 阶段映射到真实 NPU Timeline。

[返回仓库首页](../../README.md)
