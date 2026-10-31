# LLM Inference Notes

大模型推理工程文档：原理、源码、部署与性能分析。

Engineering notes on LLM inference, with an emphasis on SGLang and Ascend NPU.

[在线阅读](https://akiyangdev.github.io/llm-inference-notes/)

## 内容导航

| 主题 | 内容范围 |
| --- | --- |
| [推理基础](docs/fundamentals/README.md) | 工作知识地图、Tensor、Attention、Prefill / Decode、KV Cache |
| [SGLang 源码](docs/sglang/README.md) | 请求链路、Scheduler、批次组织与模型执行 |
| [SGLang PR 精读](docs/pr-reviews/README.md) | 从真实 merged PR 学设计、Bug、性能优化与工程取舍 |
| [Ascend 部署](docs/ascend/README.md) | 环境配置、模型部署、算子后端与排障 |
| [分布式推理](docs/distributed/README.md) | TP / DP / EP / PP、通信与数据归属 |
| [投机解码](docs/speculative-decoding/README.md) | Draft / Verify、接受逻辑与缓存状态 |
| [性能分析](docs/performance/README.md) | TTFT、TPOT、吞吐、显存与 Profiling |

## 专题文章

文章按主要阅读角色组织；专题、标签与系列阅读仍在网站中并行保留。

- [AISBench 的 whole、native、TPOT 和 ITL，到底在量什么？](docs/performance/aisbench-whole-native-tpot-itl.md)：从单请求时间线、事件间隔到原生稳定段选择，讲清延迟、并发和吞吐的样本与分母。

- [DeepEP 收到了多少路由，不等于模型真正执行了多少有效路由](docs/performance/deepep-received-vs-valid-routes.md)：结合两卡实测，解释 graph padding 如何混入接收计数，以及有效 TopK 统计怎样还原专家需求。
- [复用 TP 通信组到底省在哪里？一次 WORLD 与 HCCL 显存分配追踪](docs/performance/ascend-eplb-tp-world-memory.md)：从 OFF 后续预热和首次 WORLD 通信，追踪复用 TP 避免的额外显存分配。
- [一次 Ascend EPLB 显存优化：为什么逐层处理，仍然堆出了多层临时张量？](docs/performance/ascend-eplb-memory-bounded-staging.md)：解释逐层迁移中的跨 stream 临时张量积压，以及完成事件如何降低缓存水位。
- [静态 EPLB 之后：动态专家负载均衡怎样安全地迁移权重？](docs/distributed/ascend-dynamic-eplb-weight-migration.md)：从专家槽位、W8A8/NZ暂存和通信等待，讲到逐层路由提交、验证范围与迁移成本。

- [SGLang 流式压测：怎样保留接受率和缓存计数，又不改坏计时](docs/performance/sglang-stream-acceptance-cache-timing.md)：沿真实缺字段诊断、原SSE计时与observer写盘链，说明计数来源、事件扰动和统计窗口怎样对应。

- [128K 已经命中 90% 前缀缓存，为什么推理仍然不轻松？](docs/performance/128k-prefix-cache-remaining-costs.md)：解释前缀命中之后的新输入计算、压缩历史访问、PD状态交接和持续生成成本。

- [用 AISBench 压测 SGLang：从第一条请求到一份可信报告](docs/performance/aisbench-sglang-first-request-to-report.md)：从现有服务和一条请求开始，沿独立配置、小测、正式负载与原生稳态后处理读懂报告。

- [专家计数都采到了，为什么导出还会失败？](docs/performance/expert-count-export-hccl-gloo.md)：沿一次HCCL导出故障，解释异步错误、CPU/Gloo整数归约、真实多worker验证和循环计数的使用边界。

- [128K 输入、90% 前缀缓存：怎样把压测场景真正跑准](docs/performance/128k-prefix-cache-benchmark-calibration.md)：从首轮实际命中不足九成，到多 DP 预热、逐条验收和成本记录，展开 128K/1K 缓存场景的校准过程。

### 流量入口

面向第一次进入 AI Infra / SGLang 的读者，从更广泛的问题切入。

- [Prefill 与 Decode 为什么性能不同：从计算形状到执行开销](docs/fundamentals/prefill-vs-decode-performance.md)：从 DeepSeek-V4 的真实 Dense / MoE shape、SWA/C4/C128 长上下文读流量和 Ascend 910C Profiling 指标解释两阶段为何落在不同 Workload Shape 空间。
- [从一次 Decode 读懂 SGLang：Tensor、缓存与下一枚 Token](docs/fundamentals/sglang-decode-source-reading-primer.md)：沿普通 Decode 的 L→L+1 时间线串起 ForwardBatch、mHC、DSV4 KV ownership、MoE、Vocab Parallel Logits 与 Sampling，作为进入 SGLang 深层源码的桥梁。
- [模型怎样跨多卡运行：从显存容量到 TP、DP Attention 与 EP](docs/distributed/why-large-models-need-multi-card-tp-dp-ep.md)：从 304B / 13B 的 MoE 参数账和 910C HBM 约束出发，区分 TP 的 Tensor 分片、普通 DP 与 DPA 的请求 / KV 布局、EP 的 Expert ownership，并用 TP16 / DP8 / EP16 串起 Attention 与 MoE 的两套并行视角。
- [SGLang 请求生命周期：从 Chat Completion 到 Ascend NPU](docs/sglang/sglang-ascend-request-lifecycle.md)：以 DeepSeek-V4 为例，连接请求调度、共享与压缩缓存、Ascend 算子及流式输出，并说明 DSPARK 的验证边界。

### 进阶

- [CP2 替代 CP4：通信组变小了，为什么整体吞吐没有提高？](docs/performance/ascend-910c-cp2-vs-cp4-prefill-throughput.md)：沿CP/DP重组、每rank行数与normal分轮，解释TPOT下降但同负载native/whole吞吐未提高，以及阶段计时的归因边界。

- [NPU 时间线里有 19% 空隙，为什么不能直接得到 19% 加速？](docs/performance/npu-timeline-gaps-and-recoverable-speedup.md)：用真实rank0/1区间统计与小拷贝定位，区分记录空隙、跨rank等待、采集扰动与可回收关键路径时间。

- [Prefill chunk 应该怎么选：从 DP 折算、页对齐到通信轮次与尾块合批](docs/performance/deepseek-v4-prefill-chunk-size-selection.md)：用8192→8320与CP2负例，解释有效chunk、页预算、通信分轮和跨请求合批怎样共同决定吞吐。

- [DeepSeek-V4 IndexCache freq2：跨层 TopK 复用省了什么，又改变了什么](docs/performance/deepseek-v4-indexcache-freq2-cross-layer-topk-reuse.md)：解释freq2按c4层复用历史位置选择，结合源码与同负载对照说明少算了什么、为何仍需质量验收。

- [Ascend 910C上的P端场景静态EPLB：从路由计数、专家布局到实测收益](docs/performance/ascend-910c-prefill-scenario-static-eplb.md)：从P端路由计数生成固定专家布局，区分放置、副本分发与在线重排，并用16K/128K对照解释收益。
- [Ascend 910C推理调优负例分析：为什么合理的优化没有带来吞吐收益](docs/performance/ascend-910c-inference-tuning-negative-results-and-stop-rules.md)：从TPOT降低却吞吐不增、whole升高但native不变等对照，分析负例能支持什么结论、何时应停止试验。
- [一次Ascend 910C推理调优复盘：真正有效的优化、失败假设与实验成本](docs/performance/ascend-910c-inference-tuning-effective-optimizations-and-costs.md)：复盘P合批带来的32.27%稳态吞吐提升，以及Compact、静态布局与实验成本的取舍。
从概念进入实现，连接 Scheduler、ModelRunner、Attention、MoE、分布式与量化执行。

- [Decode 已经很快了，吞吐为什么上不去？](docs/performance/decode-fast-throughput-limited-by-prefill.md)：从 8.7ms TPOT 和 4-token 尾块追到 P 端合批，展开同负载 native 吞吐提高 32.27% 的实测过程。
- [让 AI 连续调优几天后，我发现实验流程也有瓶颈](docs/performance/inference-tuning-workflow-postmortem.md)：回看复制脚本、反复检查、采集失败和多代理等待，追问为什么忙了很久，正式实验还没发出。
- [把性能目标交给 AI 之后：大模型推理的全自动调优闭环](docs/performance/ai-driven-inference-autotuning-decision-loop.md)：沿 P 端尾块合批与并发取舍，讲清下一轮实验怎么选、什么结果值得保留。
- [一轮实验什么时候算完成：全自动调优的状态、证据与恢复](docs/performance/ai-driven-inference-autotuning-execution-and-evidence.md)：跟着配置、进程、缓存和原始统计走完整条执行链，讨论出错后从哪里接。
- [从缓存校准到并发取舍：DeepSeek-V4-Pro 在 Ascend 910C 上的一次 32k/1k 调优记录](docs/performance/deepseek-v4-pro-32k-prefix-cache-concurrency-tuning.md)：从 85.8%→91.8% 的缓存校准，到 C24/C36/C72 的吞吐与等待时间对照，解释 TPOT 的非单调变化和压测日志的统计口径。
- [GSM8K 精度排查：从 200 题失败到全量与逐题对照](docs/sglang/gsm8k-accuracy-debugging-from-200-to-full-evaluation.md)：记录 DeepSeek-V2-Lite-W8A8 的配置对照、原始评测器核对与两轮全量结果，解释为什么准确率过线和逐题输出稳定需要分别判断。
- [DeepSeek-V4 W8A8 执行链：从量化权重到 Ascend INT8 算子](docs/ascend/deepseek-v4-w8a8-ascend-910c.md)：严格区分官方 `W8A8_DYNAMIC` 主路径与静态 W8A8 对照路径，追踪 Dense QuantMatmul、MoE GroupedMatmul、DeepEP INT8 wire，以及 FRACTAL_NZ → `aclnnQuantMatmulWeightNz` → CANN QuantBatchMatmulV3。
- [SGLang Scheduler：请求怎样组批、执行和持续生成](docs/sglang/sglang-scheduler-source-analysis.md)：从 `Req`、`ScheduleBatch`、`ForwardBatch` 三个生命周期切入，追踪 admission、Prefix Cache、Prefill / Decode 与 ModelRunner 边界。
- [SGLang ModelRunner：九个输入 Token 怎样产生两行 Logits](docs/sglang/sglang-modelrunner-source-analysis.md)：沿 `ForwardBatch → ModelRunner → DeepSeek-V4 → LogitsProcessor → Sampling` 追踪一次 Prefill / Decode 的真实执行边界。
- [DeepSeek-V4 Attention：Query、缓存写入与历史寻址](docs/sglang/deepseek-v4-attention-source-analysis.md)：追踪 Query 构造、NPU cache write、request-to-page 映射，以及按层选择的 SWA / C4 / C128 历史读取路径。
- [DeepSeek-V4 MoE：Token 从路由到专家计算，再回到原位置](docs/sglang/deepseek-v4-moe-source-analysis.md)：从 Router / HashTopK 追到 Expert ownership、EP dispatch、Expert compute、combine，并解释 Ascend FuseEP 的融合边界。
- [DeepSeek-V4 分布式 Forward：Attention 与 MoE 怎样交换布局](docs/sglang/deepseek-v4-distributed-parallel-source-analysis.md)：解释 `dp_size → attn_dp/attn_tp`、Attention→MoE 布局桥、A2A 下 EP=TP，以及 TP32/DP16 在 DSpark 路径中的显式限制。
- [DeepSeek-V4 DSpark：一轮 Draft、Verify、Accept 与 Commit](docs/speculative-decoding/deepseek-v4-speculative-decoding-source-analysis.md)：沿 DSparkWorkerV2 追踪 Draft Block、Verify Planner、Target Verify、greedy/sampling accept、commit_lens 与 Target hidden → Draft state 提交。
- [DeepSeek-V4 MTP：NextN 怎样逐步生成候选](docs/speculative-decoding/deepseek-v4-mtp-speculative-decoding-source-analysis.md)：区分 MTP / NextN / EAGLE 三层语义，追踪 Target mHC hidden → Draft Extend → multi-step NextN candidate chain，并解释 Ascend NPU 的 dsv4 multi-step Draft 路径。
- [投机解码的性能收支：接受长度、Verify 成本与并发](docs/performance/deepseek-v4-speculative-performance-analysis.md)：从 Accept Length/Step Time、strict Accept Rate、DSpark SPS cost model 到 Ragged Verify，解释 speculative latency/throughput 的 break-even 与动态 Verify Budget。
- [DeepSeek-V4 PD 分离：首 Token、KV 状态与 Decode 接管](docs/sglang/deepseek-v4-pd-disaggregation-source-analysis.md)：沿 Prefill bootstrap、Decode prealloc、KV/state transfer、PREBUILT handoff 与首轮 Decode 追踪一次 P→D ownership 迁移，并拆清 Ascend DSV4 的 C4/SWA/C128 wire layout。

### 深水

源码、性能、投机解码、Ascend Profiling 与 PR 级工程细节。

- [日志里的 TP16 为什么应是 TP8：SGLang PR #39871](docs/pr-reviews/sglang-pr-39871-attention-parallel-widths.md)：从一个真实日志 Bug 切入，解释 `tp_size` 与 `attn_tp_size` 为什么不同，并把 Config Value、Runtime Derived Value、Rank / Group 与 Single Source of Truth 串成一条完整工程链路。
- [小 Tensor 为什么保活大内存：SGLang PR #39120 的 View 缓存修复——从 SGLang PR #39120 理解 View、Storage 与 Cache Memory Accounting](docs/pr-reviews/sglang-pr-39120-tensor-view-storage-cache.md)：从多模态 embedding cache 的 retained Storage 问题切入，解释为什么逻辑 Tensor payload 不等于实际保活内存，以及 ownership / lifetime 为什么必须进入 Cache 账本。
- [DeepSeek-V4 W8A8 性能分析：小 M、NZ 布局与测量方法](docs/performance/deepseek-v4-w8a8-ascend-910c-performance.md)：从 DynamicQuant 的 `coreNum=min(vectorCoreNum,M)`、QuantBatchMatmulV3 SmallMN/StreamK、真实 W8A8 Dense shape 与 GroupedMatmul `M_e` 调优边界解释 INT8 理论优势为何不一定等于端到端加速。
- [DeepSeek-V4 DSpark 的并行限制与 Draft 布局](docs/distributed/deepseek-v4-dspark-parallel-layout-analysis.md)：区分 Dense/MoE draft 的并行上下文，解释 DP/MoE token-count metadata、SpecTpSync、Verify layout，并把 TP32/DP16 的已知事实与待验证根因分开。
- [EAGLE 候选树：从候选池到 Verify 与接受路径](docs/speculative-decoding/deepseek-v4-eagle-draft-tree-source-analysis.md)：从累计路径 score、candidate-pool 索引与 parent metadata 出发，解释多分支候选如何被裁剪并恢复成 Verify Tree。
- [EAGLE Tree Attention：候选布局、位置与可见性](docs/speculative-decoding/deepseek-v4-tree-attention-source-analysis.md)：从 Candidate Layout、Tree Position、FULL_MASK / QLEN_ONLY 到 backend 消费方式，解释 multi-branch Verify 的 ancestor-only visibility。
- [SGLang 投机解码调度：请求进度、设备状态与跨轮交接](docs/speculative-decoding/sglang-speculative-decoding-scheduler-source-analysis.md)：从 Req/ReqKvInfo、ScheduleBatch、ForwardBatch、GenerationBatchResult 到 FutureMap，追一轮 Draft→Verify→Accept，重点解释 overlap 多时钟、publish fence、forward isolation 与 mixed-tail late binding。
- [DSpark 自适应 Verify：怎样分配候选预算](docs/speculative-decoding/deepseek-v4-dspark-adaptive-verify-scheduler.md)：严格区分 DeepSpec 的 soft accept-rate 训练 target 与 SGLang STS 的 greedy prefix label，解释 `tau=B+sum(survival)` 的 committed-progress 边界、1D/Additive SPS 的 Graph tier cliff 风险，以及 Ascend 910C 当前 static→compact 的真实 backend 缺口。
- [Ascend 910C 投机解码：Graph、多流与通信的执行依赖](docs/performance/deepseek-v4-speculative-ascend-performance-critical-path-analysis.md)：从 EAGLE Draft Graph、Target Verify request bucket、Plan Stream、Target Attention Multi-Stream 到 DeepEP async event，拆解 Ascend 910C 上一轮 speculative iteration 的真实 critical path，并核清 Draft-Extend Graph 的配置边界。
- [在 Ascend 910C 上分析 DSpark：从阶段计时到 NPU 时间线](docs/performance/deepseek-v4-speculative-decoding-ascend-profiling.md)：严格核清 DSpark NPU Event 计时、Ascend 上 eager Accept/Commit、DSV4 Graph replay 边界，以及 DeepEP/ZBAL async/recv-hook 的真实 overlap 能力，把源码阶段映射到 910C Profiling 时间线。
- [投机解码 Accept：从 Verify 行到最终输出路径](docs/speculative-decoding/deepseek-v4-accept-decision-source-analysis.md)：区分 Draft path score、Greedy Tree Match、Target-only Tree Sampling 与 classic p/q rejection，并解释 accept_index、bonus 与 DSpark correct_len 的真实语义。
- [投机解码 KV 管理：接受路径怎样提交、整理与回收](docs/speculative-decoding/deepseek-v4-speculative-kv-commit-compact-rollback-source-analysis.md)：追踪 accept_index 到 candidate KV、move_kv_cache(tgt, src)、accepted-path compaction、Spec V2 overshoot 回收，并拆分 Ascend DSV4 的 C4/C128 KV 与 compressor-state 生命周期。
- [多卡投机解码：Draft、Verify 与 Accept 怎样保持一致](docs/distributed/deepseek-v4-speculative-distributed-consistency-source-analysis.md)：拆分 speculative control、logical layout 与 model-data collective 三层一致性，对比 DSpark SpecTpSync 与 EAGLE Accept broadcast，并厘清 Ascend HCCL、ZBAL、DeepEP 的职责边界。

### Reference

适合反复查阅的名词表、并行拓扑、通信语义与知识地图。

- [AI 推理基础设施名词表：从 Token 到 Ascend Kernel](docs/fundamentals/ai-infra-working-glossary.md)：作为全仓库总入口，串起模型、Runtime、内存、分布式、投机解码与 Ascend 执行层，并提供当前 SGLang 源码入口索引和后续专题阅读路线。
- [SGLang Rank 与 Group：读懂 TP、DP Attention 和 EP 坐标](docs/distributed/sglang-rank-group-deepseek-v4.md)：作为分布式源码前置，区分 Native DP 与 DPA 下的 `dp_rank`，解释 Attention 的 `DP × CP × TP`、MoE 的 `DP × EP × TP` 坐标，以及 Ascend 910C 上独立 `_MOE_EP` Group。
- [SGLang Collective：从张量分片到 Attention 与 MoE 通信](docs/distributed/sglang-collectives-deepseek-v4.md)：沿 Tensor ownership 解释 TP AllReduce/AllGather、DPA 的三类 gather、MoE ReduceScatter，以及 Ascend DeepEP / FuseEP 的 A2A 数据面边界。

## 配套内容

- [代码示例](examples/README.md)：文章配套的可运行示例。
- [图表资源](assets/README.md)：架构图、调用链图和实验图表。
- [写作模板](templates/README.md)：源码解析与部署实践模板。
- [AI Infra Skills](skills/README.md)：源码取证、SGLang PR 精读、连续工程叙事、图解与文章评审规则。
- [贡献与维护](CONTRIBUTING.md)：命名、证据要求与检查方式。

文章按具体版本解释实现；原理示例、源码确认与实测结果分别标注。
