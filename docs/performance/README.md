# 性能分析

时延、吞吐、显存、算子、通信与 Profiling。这里重点回答的不只是“哪个 Kernel 最慢”，而是如何从源码建立性能假设，再用 Ascend 的真实运行数据与 Timeline 验证。

## 调优实践

- [SGLang 流式压测：怎样保留接受率和缓存计数，又不改坏计时](sglang-stream-acceptance-cache-timing.md)：附原解析方法的离线回放，解释metadata-only事件、父类后保留字段、缺失计数与AL/接受率分母。

- [128K 已经命中 90% 前缀缓存，为什么推理仍然不轻松？](128k-prefix-cache-remaining-costs.md)：区分token命中、计算量、状态驻留和传输，结合真实128K实验解释剩余开销。

- [用 AISBench 压测 SGLang：从第一条请求到一份可信报告](aisbench-sglang-first-request-to-report.md)：附可运行配置、JSONL与脚本，解释模型名、tokenizer、N/C/r、retry及whole/native结果保存。

- [CP2 替代 CP4：通信组变小了，为什么整体吞吐没有提高？](ascend-910c-cp2-vs-cp4-prefill-throughput.md)：沿CP/DP重组、每rank行数与normal分轮，解释TPOT下降但同负载native/whole吞吐未提高，以及阶段计时的归因边界。

- [专家计数都采到了，为什么导出还会失败？](expert-count-export-hccl-gloo.md)：从storage.cpu暴露的HcclAllreduce内存申请失败，走到同成员CPU/Gloo导出和有效统计文件验收。

- [NPU 时间线里有 19% 空隙，为什么不能直接得到 19% 加速？](npu-timeline-gaps-and-recoverable-speedup.md)：用真实rank0/1区间统计与小拷贝定位，区分记录空隙、跨rank等待、采集扰动与可回收关键路径时间。

- [128K 输入、90% 前缀缓存：怎样把压测场景真正跑准](128k-prefix-cache-benchmark-calibration.md)：沿真实 tokenizer 核长、共享区与独立后缀、seed/check 和正式 usage，解释预测前缀与实际命中的差别。

- [Prefill chunk 应该怎么选：从 DP 折算、页对齐到通信轮次与尾块合批](deepseek-v4-prefill-chunk-size-selection.md)：用8192→8320与CP2负例，解释有效chunk、页预算、通信分轮和跨请求合批怎样共同决定吞吐。

- [DeepSeek-V4 IndexCache freq2：跨层 TopK 复用省了什么，又改变了什么](deepseek-v4-indexcache-freq2-cross-layer-topk-reuse.md)：沿c4层调度、compressor更新和Graph共享存储，解释同负载native吞吐提升5.41%的来源与精度边界。

- [Ascend 910C上的P端场景静态EPLB：从路由计数、专家布局到实测收益](ascend-910c-prefill-scenario-static-eplb.md)：沿逻辑专家、副本分发、循环计数与离线拟合，解释16K方案whole提升14.18%的归因边界和128K小收益。
- [Decode 已经很快了，吞吐为什么上不去？](decode-fast-throughput-limited-by-prefill.md)：沿请求供给、分块接纳和页对齐，解释 Pcap1→2 为何提高吞吐而 TPOT 略高。
- [Ascend 910C推理调优负例分析：为什么合理的优化没有带来吞吐收益](ascend-910c-inference-tuning-negative-results-and-stop-rules.md)：逐项分析CP/chunk、到达速率、Compact与SPS预算、静态EPLB的小收益和负例，区分瓶颈假设、窗口差异与停止条件。
- [一次Ascend 910C推理调优复盘：真正有效的优化、失败假设与实验成本](ascend-910c-inference-tuning-effective-optimizations-and-costs.md)：从P端尾块合批、IndexCache与静态布局的实测收益，讲到Compact负例、缓存口径和实验成本。
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
