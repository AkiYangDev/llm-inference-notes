# 投机推理

Draft / Verify、接受策略、Verify 调度、缓存提交与多卡一致性。这里重点关注 DeepSeek-V4 与 SGLang 中 speculative state 如何真正进入 Runtime，而不是只解释算法概念。

## DSpark 自适应 Verify

- [DeepSeek-V4 DSpark 到底怎么决定 Verify 多少 Token？从 Confidence Head、STS Calibration 到 SPS Cost Model 与 Ragged Verify](deepseek-v4-dspark-adaptive-verify-scheduler.md)：从 DeepSpec 的 Confidence Head 训练语义追到 SGLang STS、expected-progress surrogate、SPS cost model、per-request verify_lens，并严格分析 Ascend 910C 为什么当前仍固定 static、放开 compact 需要补哪些 NPU backend / graph / DP 能力。

## 配套阅读

- [DSpark 一轮 Draft → Verify → Accept/Reject](../sglang/deepseek-v4-speculative-decoding-source-analysis.md)
- [Speculative Scheduler 状态机](../sglang/sglang-speculative-decoding-scheduler-source-analysis.md)
- [Speculative Decoding 性能模型](../sglang/deepseek-v4-speculative-performance-analysis.md)
- [Ascend 910C Critical Path](../sglang/deepseek-v4-speculative-ascend-performance-critical-path-analysis.md)
- [Ascend 910C Profiling 指南](../performance/deepseek-v4-speculative-decoding-ascend-profiling.md)

[返回仓库首页](../../README.md)
