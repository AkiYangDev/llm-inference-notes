# 性能分析

时延、吞吐、显存、算子、通信与 Profiling。这里重点回答的不只是“哪个 Kernel 最慢”，而是如何从源码建立性能假设，再用 Ascend 的真实运行数据与 Timeline 验证。

## 调优实践

- [让 AI 连续调优几天后，我发现实验流程也有瓶颈](inference-tuning-workflow-postmortem.md)：沿真实时间账与几次返工，讨论脚本复制、检查错位、多代理协作和跨会话接续。
- [把性能目标交给 AI 之后：大模型推理的全自动调优闭环](ai-driven-inference-autotuning-decision-loop.md)：从固定场景、瓶颈证据到短筛确认，讨论下一轮实验怎么选。
- [一轮实验什么时候算完成：全自动调优的状态、证据与恢复](ai-driven-inference-autotuning-execution-and-evidence.md)：沿一次实验的执行和采集过程，说明热状态接续、计数范围与跨会话交接。
- [Ascend 950DT 单机 PD 调优：128K/1K、90% Cache 场景下的配置与实测](ascend-950dt-single-node-pd-optimization.md)：从整体结果展开 P 端缓存调整、D 端 DSPARK、dispatch 容量与图档位，结合四张图和 N192/C48 完整实测解释优化效果。
- [从缓存校准到并发取舍：DeepSeek-V4-Pro 在 Ascend 910C 上的一次 32k/1k 调优记录](deepseek-v4-pro-32k-prefix-cache-concurrency-tuning.md)：记录共享前缀命中率校准、C24/C36/C72 的同 N 对照，结合 TTFT、TPOT、输出吞吐与客户端统计口径分析取舍；附实验 CSV 和测试脚本。

## 性能直觉入口

- [Prefill 与 Decode 为什么性能不同：从计算形状到执行开销](../fundamentals/prefill-vs-decode-performance.md)：先用 DeepSeek-V4 的 Dense M、Expert M_e、SWA/C4/C128 与 Ascend 910C PMU 指标建立性能分析坐标，再进入具体算子和 Profiling。

## W8A8 与算子性能

- [DeepSeek-V4 W8A8 性能分析：小 M、NZ 布局与测量方法](deepseek-v4-w8a8-ascend-910c-performance.md)：分析 DynamicQuant 的小 M 并行度、QuantBatchMatmulV3 的 SmallMN / StreamK tiling、DeepSeek-V4 实际 FRACTAL_NZ shape，以及 GroupedMatmul 的 expert-token 调优边界。

## 投机解码性能

- [投机解码的性能收支：接受长度、Verify 成本与并发](deepseek-v4-speculative-performance-analysis.md)：从 Accept Length / Step Time、DSpark SPS cost model 到 Ragged Verify，建立 speculative decoding 的 break-even 模型。
- [Ascend 910C 投机解码：Graph、多流与通信的执行依赖](deepseek-v4-speculative-ascend-performance-critical-path-analysis.md)：拆解 Draft Graph、Target Verify、Plan Stream、Target Attention Multi-Stream 与 DeepEP async event 的 critical path。
- [在 Ascend 910C 上分析 DSpark：从阶段计时到 NPU 时间线](deepseek-v4-speculative-decoding-ascend-profiling.md)：把 DSpark Draft / Verify、NPU Graph、Accept / Commit 与 DeepEP/ZBAL 映射到真实 910C Timeline。

## 建议阅读顺序

1. W8A8 方向先读 [执行链](../ascend/deepseek-v4-w8a8-ascend-910c.md)，再读 [W8A8 性能分析](deepseek-v4-w8a8-ascend-910c-performance.md)。
2. 投机解码方向先读 [执行链与状态机](../speculative-decoding/README.md)，再读 [性能模型](deepseek-v4-speculative-performance-analysis.md)。
3. 最后进入 [Ascend 910C Critical Path](deepseek-v4-speculative-ascend-performance-critical-path-analysis.md) 与 [Profiling 指南](deepseek-v4-speculative-decoding-ascend-profiling.md)。

[返回仓库首页](../../README.md)
