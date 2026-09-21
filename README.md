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

- [AI 推理基础设施工作名词表：SGLang、DeepSeek 与 Ascend 910C 从 Token 到 NPU Kernel](docs/fundamentals/ai-infra-working-glossary.md)：作为全仓库总入口，串起模型、Runtime、内存、分布式、投机解码与 Ascend 执行层，并提供当前 SGLang 源码入口索引和后续专题阅读路线。
- [为什么 Prefill 和 Decode 明明跑的是同一个模型，性能却完全不同？](docs/fundamentals/prefill-vs-decode-performance.md)：从 DeepSeek-V4 的真实 Dense / MoE shape、SWA/C4/C128 长上下文读流量和 Ascend 910C Profiling 指标解释两阶段为何落在不同 Workload Shape 空间。
- [第一次读 SGLang 源码，应该先看懂什么？用 DeepSeek-V4 一次 Decode 串起 Tensor、KV Cache、Attention、MoE 与 Sampling](docs/fundamentals/sglang-decode-source-reading-primer.md)：沿普通 Decode 的 L→L+1 时间线串起 ForwardBatch、mHC、DSV4 KV ownership、MoE、Vocab Parallel Logits 与 Sampling，作为进入 SGLang 深层源码的桥梁。
- [模型为什么必须多卡？从一张 Ascend 910C 放不下 DeepSeek-V4 到 TP / DP / EP](docs/distributed/why-large-models-need-multi-card-tp-dp-ep.md)：从 304B / 13B 的 MoE 参数账和 910C HBM 约束出发，区分 TP 的 Tensor 分片、普通 DP 与 DPA 的请求 / KV 布局、EP 的 Expert ownership，并用 TP16 / DP8 / EP16 串起 Attention 与 MoE 的两套并行视角。
- [SGLang 里的 Rank 和 Group 到底是什么？用 DeepSeek-V4 画清 TP Rank、DP Rank、EP Rank](docs/distributed/sglang-rank-group-deepseek-v4.md)：作为分布式源码前置，区分 Native DP 与 DPA 下的 `dp_rank`，解释 Attention 的 `DP × CP × TP`、MoE 的 `DP × EP × TP` 坐标，以及 Ascend 910C 上独立 `_MOE_EP` Group。
- [从一个 10 行 PR 看懂 SGLang 的 Attention 并行拓扑：PR #39871 为什么 TP16 不是 AttnTP16？](docs/pr-reviews/sglang-pr-39871-attention-parallel-widths.md)：从一个真实日志 Bug 切入，解释 `tp_size` 与 `attn_tp_size` 为什么不同，并把 Config Value、Runtime Derived Value、Rank / Group 与 Single Source of Truth 串成一条完整工程链路。
- [一个 Tensor 明明很小，为什么却偷偷占着整个 Batch？——从 SGLang PR #39120 理解 View、Storage 与 Cache Memory Accounting](docs/pr-reviews/sglang-pr-39120-tensor-view-storage-cache.md)：从多模态 embedding cache 的 retained Storage 问题切入，解释为什么逻辑 Tensor payload 不等于实际保活内存，以及 ownership / lifetime 为什么必须进入 Cache 账本。
- [SGLang 里的 AllReduce、AllGather、ReduceScatter、All-to-All 到底在搬什么？用 DeepSeek-V4 画清 Group、Tensor 和通信方向](docs/distributed/sglang-collectives-deepseek-v4.md)：沿 Tensor ownership 解释 TP AllReduce/AllGather、DPA 的三类 gather、MoE ReduceScatter，以及 Ascend DeepEP / FuseEP 的 A2A 数据面边界。
- [DeepSeek-V4 W8A8 推理在 Ascend 910C 上到底发生了什么？从量化权重到 INT8 MatMul Kernel](docs/ascend/deepseek-v4-w8a8-ascend-910c.md)：严格区分官方 `W8A8_DYNAMIC` 主路径与静态 W8A8 对照路径，追踪 Dense QuantMatmul、MoE GroupedMatmul、DeepEP INT8 wire，以及 FRACTAL_NZ → `aclnnQuantMatmulWeightNz` → CANN QuantBatchMatmulV3。
- [DeepSeek-V4 W8A8 在 Ascend 910C 上为什么不一定更快？从 Decode 小 M、FRACTAL_NZ 到 CANN Tiling](docs/performance/deepseek-v4-w8a8-ascend-910c-performance.md)：从 DynamicQuant 的 `coreNum=min(vectorCoreNum,M)`、QuantBatchMatmulV3 SmallMN/StreamK、真实 W8A8 Dense shape 与 GroupedMatmul `M_e` 调优边界解释 INT8 理论优势为何不一定等于端到端加速。
- [SGLang 推理执行链路解析：一次 Chat Completion 请求是如何跑到 Ascend NPU 上的](docs/sglang/sglang-ascend-request-lifecycle.md)：以 DeepSeek-V4 为例，连接请求调度、共享与压缩缓存、Ascend 算子及流式输出，并说明 DSPARK 的验证边界。
- [SGLang Scheduler 源码解析：一个请求是如何被组批、调度并送进 ModelRunner 的](docs/sglang/sglang-scheduler-source-analysis.md)：从 `Req`、`ScheduleBatch`、`ForwardBatch` 三个生命周期切入，追踪 admission、Prefix Cache、Prefill / Decode 与 ModelRunner 边界。
- [SGLang ModelRunner 源码解析：ForwardBatch 如何驱动 DeepSeek 完成一次 Prefill / Decode](docs/sglang/sglang-modelrunner-source-analysis.md)：沿 `ForwardBatch → ModelRunner → DeepSeek-V4 → LogitsProcessor → Sampling` 追踪一次 Prefill / Decode 的真实执行边界。
- [DeepSeek-V4 Attention 源码解析：一个 ForwardBatch 如何变成 Query，并通过 KV Cache 找回历史上下文](docs/sglang/deepseek-v4-attention-source-analysis.md)：追踪 Query 构造、NPU cache write、request-to-page 映射，以及按层选择的 SWA / C4 / C128 历史读取路径。
- [DeepSeek-V4 MoE 推理源码解析：一个 Token 如何经过 Router、All-to-All 和 Expert Parallel 完成一次前向计算](docs/sglang/deepseek-v4-moe-source-analysis.md)：从 Router / HashTopK 追到 Expert ownership、EP dispatch、Expert compute、combine，并解释 Ascend FuseEP 的融合边界。
- [DeepSeek-V4 分布式推理源码解析：TP、EP、DP 三种并行如何共同驱动一次前向计算](docs/sglang/deepseek-v4-distributed-parallel-source-analysis.md)：解释 `dp_size → attn_dp/attn_tp`、Attention→MoE 布局桥、A2A 下 EP=TP，以及 TP32/DP16 在 DSpark 路径中的显式限制。
- [DeepSeek-V4 DSpark 源码解析：Draft Model 为什么需要独立处理 TP/DP/EP Layout？](docs/distributed/deepseek-v4-dspark-parallel-layout-analysis.md)：区分 Dense/MoE draft 的并行上下文，解释 DP/MoE token-count metadata、SpecTpSync、Verify layout，并把 TP32/DP16 的已知事实与待验证根因分开。
- [DeepSeek-V4 Speculative Decoding 源码解析：Draft → Verify → Accept/Reject 如何驱动一次投机解码](docs/speculative-decoding/deepseek-v4-speculative-decoding-source-analysis.md)：沿 DSparkWorkerV2 追踪 Draft Block、Verify Planner、Target Verify、greedy/sampling accept、commit_lens 与 Target hidden → Draft state 提交。
- [DeepSeek-V4 Speculative Decoding 源码解析：MTP（Multi-Token Prediction）如何让 Draft 一次预测多个 Token](docs/speculative-decoding/deepseek-v4-mtp-speculative-decoding-source-analysis.md)：区分 MTP / NextN / EAGLE 三层语义，追踪 Target mHC hidden → Draft Extend → multi-step NextN candidate chain，并解释 Ascend NPU 的 dsv4 multi-step Draft 路径。
- [DeepSeek-V4 Speculative Decoding 源码解析：EAGLE Draft Tree 如何组织多分支候选，并驱动一次 Verify](docs/speculative-decoding/deepseek-v4-eagle-draft-tree-source-analysis.md)：从累计路径 score、candidate-pool 索引与 parent metadata 出发，解释多分支候选如何被裁剪并恢复成 Verify Tree。
- [DeepSeek-V4 Speculative Decoding 源码解析：Verify 为什么需要 Tree Attention？Tree Mask、Position 与 Candidate Layout 如何协同](docs/speculative-decoding/deepseek-v4-tree-attention-source-analysis.md)：从 Candidate Layout、Tree Position、FULL_MASK / QLEN_ONLY 到 backend 消费方式，解释 multi-branch Verify 的 ancestor-only visibility。
- [SGLang Speculative Decoding Scheduler 源码解析：ScheduleBatch 如何管理 Draft、Verify、Accept 状态机？](docs/speculative-decoding/sglang-speculative-decoding-scheduler-source-analysis.md)：从 Req/ReqKvInfo、ScheduleBatch、ForwardBatch、GenerationBatchResult 到 FutureMap，追一轮 Draft→Verify→Accept，重点解释 overlap 多时钟、publish fence、forward isolation 与 mixed-tail late binding。
- [DeepSeek-V4 DSpark 到底怎么决定 Verify 多少 Token？从 Confidence Head、STS Calibration 到 SPS Cost Model 与 Ragged Verify](docs/speculative-decoding/deepseek-v4-dspark-adaptive-verify-scheduler.md)：严格区分 DeepSpec 的 soft accept-rate 训练 target 与 SGLang STS 的 greedy prefix label，解释 `tau=B+sum(survival)` 的 committed-progress 边界、1D/Additive SPS 的 Graph tier cliff 风险，以及 Ascend 910C 当前 static→compact 的真实 backend 缺口。
- [DeepSeek-V4 Speculative Decoding 性能解析：为什么 Speculative Decoding 不一定更快？从 Acceptance Rate、Draft Cost 到 KV Overhead](docs/performance/deepseek-v4-speculative-performance-analysis.md)：从 Accept Length/Step Time、strict Accept Rate、DSpark SPS cost model 到 Ragged Verify，解释 speculative latency/throughput 的 break-even 与动态 Verify Budget。
- [DeepSeek-V4 Speculative Decoding 在 Ascend 910C 上的性能源码解析：NPU Graph、Multi-Stream 与 Communication Overlap](docs/performance/deepseek-v4-speculative-ascend-performance-critical-path-analysis.md)：从 EAGLE Draft Graph、Target Verify request bucket、Plan Stream、Target Attention Multi-Stream 到 DeepEP async event，拆解 Ascend 910C 上一轮 speculative iteration 的真实 critical path，并核清 Draft-Extend Graph 的配置边界。
- [DeepSeek-V4 Speculative Decoding 在 Ascend 910C 上怎么 Profile？从 SGLang Draft/Verify 到 NPU Kernel 时间线](docs/performance/deepseek-v4-speculative-decoding-ascend-profiling.md)：严格核清 DSpark NPU Event 计时、Ascend 上 eager Accept/Commit、DSV4 Graph replay 边界，以及 DeepEP/ZBAL async/recv-hook 的真实 overlap 能力，把源码阶段映射到 910C Profiling 时间线。
- [DeepSeek-V4 Speculative Decoding 源码解析：Accept Decision 如何选择最终路径？从 Rejection Sampling 到 Parent Tree Recovery](docs/speculative-decoding/deepseek-v4-accept-decision-source-analysis.md)：区分 Draft path score、Greedy Tree Match、Target-only Tree Sampling 与 classic p/q rejection，并解释 accept_index、bonus 与 DSpark correct_len 的真实语义。
- [DeepSeek-V4 Speculative Decoding 源码解析：Draft Tree 如何落到 KV Cache，Accept 后缓存如何 Commit、Compact 与 Rollback？](docs/speculative-decoding/deepseek-v4-speculative-kv-commit-compact-rollback-source-analysis.md)：追踪 accept_index 到 candidate KV、move_kv_cache(tgt, src)、accepted-path compaction、Spec V2 overshoot 回收，并拆分 Ascend DSV4 的 C4/C128 KV 与 compressor-state 生命周期。
- [DeepSeek-V4 Speculative Decoding 源码解析：多卡场景下 Draft、Verify 与 Accept 如何保持一致](docs/distributed/deepseek-v4-speculative-distributed-consistency-source-analysis.md)：拆分 speculative control、logical layout 与 model-data collective 三层一致性，对比 DSpark SpecTpSync 与 EAGLE Accept broadcast，并厘清 Ascend HCCL、ZBAL、DeepEP 的职责边界。
- [DeepSeek-V4 Prefill / Decode Disaggregation 源码解析：为什么生产系统需要拆分 Prefill 和 Decode](docs/sglang/deepseek-v4-pd-disaggregation-source-analysis.md)：沿 Prefill bootstrap、Decode prealloc、KV/state transfer、PREBUILT handoff 与首轮 Decode 追踪一次 P→D ownership 迁移，并拆清 Ascend DSV4 的 C4/SWA/C128 wire layout。

## 配套内容

- [代码示例](examples/README.md)：文章配套的可运行示例。
- [图表资源](assets/README.md)：架构图、调用链图和实验图表。
- [写作模板](templates/README.md)：源码解析与部署实践模板。
- [AI Infra Skills](skills/README.md)：源码取证、SGLang PR 精读、连续工程叙事、图解与文章评审规则。
- [贡献与维护](CONTRIBUTING.md)：命名、证据要求与检查方式。

文章按具体版本解释实现；原理示例、源码确认与实测结果分别标注。
