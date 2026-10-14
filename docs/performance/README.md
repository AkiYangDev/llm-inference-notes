# 性能分析

时延、吞吐、显存、算子、通信与 Profiling。这里重点回答的不只是“哪个 Kernel 最慢”，而是如何从源码建立性能假设，再用 Ascend 的真实运行数据与 Timeline 验证。

## 调优实践

- [Ascend 950DT 单机 PD 调优：128K/1K、90% Cache 场景下的配置与实测](ascend-950dt-single-node-pd-optimization.md)：从整体结果展开 P 端缓存调整、D 端 DSPARK、dispatch 容量与图档位，结合四张图和 N192/C48 完整实测解释优化效果。
- [从缓存校准到并发取舍：DeepSeek-V4-Pro 在 Ascend 910C 上的一次 32k/1k 调优记录](deepseek-v4-pro-32k-prefix-cache-concurrency-tuning.md)：记录共享前缀命中率校准、C24/C36/C72 的同 N 对照，结合 TTFT、TPOT、输出吞吐与客户端统计口径分析取舍；附实验 CSV 和测试脚本。

## 性能直觉入口

- [为什么 Prefill 和 Decode 明明跑的是同一个模型，性能却完全不同？](../fundamentals/prefill-vs-decode-performance.md)：先用 DeepSeek-V4 的 Dense M、Expert M_e、SWA/C4/C128 与 Ascend 910C PMU 指标建立性能分析坐标，再进入具体算子和 Profiling。

## W8A8 与算子性能

- [DeepSeek-V4 W8A8 在 Ascend 910C 上为什么不一定更快？从 Decode 小 M、FRACTAL_NZ 到 CANN Tiling](deepseek-v4-w8a8-ascend-910c-performance.md)：严格核对 DynamicQuant 的小 M 并行度、QuantBatchMatmulV3 的 SmallMN / StreamK tiling、DeepSeek-V4 实际 FRACTAL_NZ shape，以及 GroupedMatmul 的 expert-token 调优边界。

## 投机解码性能

- [DeepSeek-V4 Speculative Decoding 性能解析：为什么 Speculative Decoding 不一定更快？从 Acceptance Rate、Draft Cost 到 KV Overhead](deepseek-v4-speculative-performance-analysis.md)：从 Accept Length / Step Time、DSpark SPS cost model 到 Ragged Verify，建立 speculative decoding 的 break-even 模型。
- [DeepSeek-V4 Speculative Decoding 在 Ascend 910C 上的性能源码解析：NPU Graph、Multi-Stream 与 Communication Overlap](deepseek-v4-speculative-ascend-performance-critical-path-analysis.md)：拆解 Draft Graph、Target Verify、Plan Stream、Target Attention Multi-Stream 与 DeepEP async event 的 critical path。
- [DeepSeek-V4 Speculative Decoding 在 Ascend 910C 上怎么 Profile？从 SGLang Draft/Verify 到 NPU Kernel 时间线](deepseek-v4-speculative-decoding-ascend-profiling.md)：把 DSpark Draft / Verify、NPU Graph、Accept / Commit 与 DeepEP/ZBAL 映射到真实 910C Timeline。

## 建议阅读顺序

1. W8A8 方向先读 [执行链](../ascend/deepseek-v4-w8a8-ascend-910c.md)，再读 [W8A8 性能源码 Review](deepseek-v4-w8a8-ascend-910c-performance.md)。
2. 投机解码方向先读 [执行链与状态机](../speculative-decoding/README.md)，再读 [性能模型](deepseek-v4-speculative-performance-analysis.md)。
3. 最后进入 [Ascend 910C Critical Path](deepseek-v4-speculative-ascend-performance-critical-path-analysis.md) 与 [Profiling 指南](deepseek-v4-speculative-decoding-ascend-profiling.md)。

[返回仓库首页](../../README.md)
