# 性能分析

时延、吞吐、显存、算子、通信与 Profiling。这里重点回答的不只是“哪个 Kernel 最慢”，而是如何从源码建立性能假设，再用 Ascend 910C 的真实 Timeline 验证。

## 专题文章

- [DeepSeek-V4 W8A8 在 Ascend 910C 上为什么不一定更快？从 Decode 小 M、FRACTAL_NZ 到 CANN Tiling](deepseek-v4-w8a8-ascend-910c-performance.md)：严格核对 DynamicQuant 的小 M 并行度、QuantBatchMatmulV3 的 SmallMN / StreamK tiling、DeepSeek-V4 实际 FRACTAL_NZ shape，以及 GroupedMatmul 的 expert-token 调优边界，区分源码能证明的机制与必须依赖 profiler 的性能结论。
- [DeepSeek-V4 Speculative Decoding 在 Ascend 910C 上怎么 Profile？从 SGLang Draft/Verify 到 NPU Kernel 时间线](deepseek-v4-speculative-decoding-ascend-profiling.md)：从 DSpark 内部 Draft/Verify Event、NPU Graph、eager Accept/Commit、DeepEP/ZBAL 到 CANN Timeline，建立源码阶段与设备时间线的严格映射，并说明哪些性能结论必须依赖真实 Trace。

## 建议阅读顺序

1. 先读 [W8A8 执行链](../ascend/deepseek-v4-w8a8-ascend-910c.md)，理解 ModelSlim、INT8 QuantMatmul、GroupedMatmul 与 CANN 边界。
2. 再读本文的 [W8A8 性能源码 Review](deepseek-v4-w8a8-ascend-910c-performance.md)，理解为什么小 M、Layout 与 Tiling 决定理论 INT8 优势能否兑现。
3. 投机推理方向先读 [Speculative Decoding 性能模型](../sglang/deepseek-v4-speculative-performance-analysis.md)，再读 [Ascend 910C Critical Path](../sglang/deepseek-v4-speculative-ascend-performance-critical-path-analysis.md)。
4. 最后进入 [Ascend 910C Profiling 指南](deepseek-v4-speculative-decoding-ascend-profiling.md)，把源码机制映射到真实 NPU Timeline。

[返回仓库首页](../../README.md)
