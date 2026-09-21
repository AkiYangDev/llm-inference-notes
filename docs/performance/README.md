# 性能分析

时延、吞吐、显存、算子、通信与 Profiling。这里重点回答的不只是“哪个 Kernel 最慢”，而是如何把 SGLang Runtime 阶段映射到 Ascend 910C 的真实 Critical Path。

## 专题文章

- [DeepSeek-V4 Speculative Decoding 在 Ascend 910C 上怎么 Profile？从 SGLang Draft/Verify 到 NPU Kernel 时间线](deepseek-v4-speculative-decoding-ascend-profiling.md)：从 DSpark 内部 Draft/Verify Event、NPU Graph、eager Accept/Commit、DeepEP/ZBAL 到 CANN Timeline，建立源码阶段与设备时间线的严格映射，并说明哪些性能结论必须依赖真实 Trace。

## 建议阅读顺序

1. 先读 [Speculative Decoding 性能模型](../sglang/deepseek-v4-speculative-performance-analysis.md)，理解 Acceptance、Step Cost 与 Effective Progress。
2. 再读 [Ascend 910C Critical Path 源码解析](../sglang/deepseek-v4-speculative-ascend-performance-critical-path-analysis.md)，理解 NPU Graph、Multi-Stream 与 Communication Overlap 的代码机制。
3. 最后进入本文 Profiling 指南，把这些源码机制映射到真实 NPU Timeline。

[返回仓库首页](../../README.md)
