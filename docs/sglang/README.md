# SGLang 源码

请求入口、调度、批次数据、模型执行和输出处理。

> 第一次接触这些概念时，建议先读 [AI 推理基础设施工作名词表](../fundamentals/ai-infra-working-glossary.md)，先建立 Token、KV Cache、Scheduler、TP / DP / EP、DSpark 到 Ascend Kernel 的整体地图，再进入源码细节。

## 专题文章

- [SGLang 推理执行链路解析：一次 Chat Completion 请求是如何跑到 Ascend NPU 上的](sglang-ascend-request-lifecycle.md)：DeepSeek-V4、`dsv4` Ascend 后端与 DSPARK 生成边界。
- [SGLang Scheduler 源码解析：一个请求是如何被组批、调度并送进 ModelRunner 的](sglang-scheduler-source-analysis.md)：从 `Req` 到 `ScheduleBatch`、`ForwardBatch`，解释 admission、Prefix Cache、Prefill / Decode 调度与 ModelRunner 边界。
- [SGLang ModelRunner 源码解析：ForwardBatch 如何驱动 DeepSeek 完成一次 Prefill / Decode](sglang-modelrunner-source-analysis.md)：解释 `ForwardBatch` 如何经过 Graph / Eager 分发进入 DeepSeek-V4，并从 packed hidden states 走到 logits 与 Sampling。
- [DeepSeek-V4 Attention 源码解析：一个 ForwardBatch 如何变成 Query，并通过 KV Cache 找回历史上下文](deepseek-v4-attention-source-analysis.md)：从 `MQALayer` 追到 Query、SWA/C4/C128 缓存、page table 与 Ascend DSV4 Attention backend。
- [DeepSeek-V4 MoE 推理源码解析：一个 Token 如何经过 Router、All-to-All 和 Expert Parallel 完成一次前向计算](deepseek-v4-moe-source-analysis.md)：解释 V4 Router / HashTopK、shared experts、EP token routing、DeepEP 与 Ascend FuseEP 的 runtime 边界。
- [DeepSeek-V4 分布式推理源码解析：TP、EP、DP 三种并行如何共同驱动一次前向计算](deepseek-v4-distributed-parallel-source-analysis.md)：从 parallel-state 派生关系追到 Attention TP、DP Attention、MoE EP、布局 split/gather 与 Ascend HCCL 边界。
- [DeepSeek-V4 DSpark 源码解析：Draft Model 为什么需要独立处理 TP/DP/EP Layout？](deepseek-v4-dspark-parallel-layout-analysis.md)：从 DSpark guard、Draft ForwardBatch、DP/MoE sync metadata、SpecTpSync 到 Verify/Commit ownership，分析 TP≠DP 的真实边界。
- [DeepSeek-V4 Speculative Decoding 源码解析：Draft → Verify → Accept/Reject 如何驱动一次投机推理](deepseek-v4-speculative-decoding-source-analysis.md)：完整追踪 DSpark 一轮 Decode，从 Draft Proposal 到 Target Verify、Accept/Reject、commit 与下一轮 Draft state。
- [DeepSeek-V4 Speculative Decoding 源码解析：MTP（Multi-Token Prediction）如何让 Draft 一次预测多个 Token](deepseek-v4-mtp-speculative-decoding-source-analysis.md)：从 bundled MTP 权重到 NextN Draft Architecture，再到 EAGLE Runtime，解释 Draft Extend、multi-step rollout、Target Verify 与 Ascend NPU dsv4 backend。
- [DeepSeek-V4 Speculative Decoding 源码解析：EAGLE Draft Tree 如何组织多分支候选，并驱动一次 Verify](deepseek-v4-eagle-draft-tree-source-analysis.md)：追踪累计路径 score、candidate-pool 索引与 parent metadata，理解多分支候选如何被裁剪并恢复成 Verify Tree。
- [DeepSeek-V4 Speculative Decoding 源码解析：Verify 为什么需要 Tree Attention？Tree Mask、Position 与 Candidate Layout 如何协同](deepseek-v4-tree-attention-source-analysis.md)：从 Candidate Layout、Tree Position、FULL_MASK / QLEN_ONLY 到 backend 消费方式，解释 multi-branch Verify 的 ancestor-only visibility。
- [SGLang Speculative Decoding Scheduler 源码解析：ScheduleBatch 如何管理 Draft、Verify、Accept 状态机？](sglang-speculative-decoding-scheduler-source-analysis.md)：从 Req/ReqKvInfo、ScheduleBatch、ForwardBatch、GenerationBatchResult 到 FutureMap，追一轮 Draft→Verify→Accept，重点解释 overlap 多时钟、publish fence、forward isolation 与 mixed-tail late binding。
- [DeepSeek-V4 DSpark 到底怎么决定 Verify 多少 Token？从 Confidence Head、STS Calibration 到 SPS Cost Model 与 Ragged Verify](../speculative-decoding/deepseek-v4-dspark-adaptive-verify-scheduler.md)：承接 Scheduler 状态机，进一步解释 Confidence/STS 如何形成 survival、SPS 如何选择 global budget，以及 Ascend 910C 当前为何仍停在 fixed-width static verify。
- [DeepSeek-V4 Speculative Decoding 性能解析：为什么 Speculative Decoding 不一定更快？从 Acceptance Rate、Draft Cost 到 KV Overhead](deepseek-v4-speculative-performance-analysis.md)：从 Accept Length/Step Time、strict Accept Rate、DSpark SPS cost model 到 Ragged Verify，解释 speculative latency/throughput 的 break-even 与动态 Verify Budget。
- [DeepSeek-V4 Speculative Decoding 在 Ascend 910C 上的性能源码解析：NPU Graph、Multi-Stream 与 Communication Overlap](deepseek-v4-speculative-ascend-performance-critical-path-analysis.md)：从 Draft Graph、固定宽度 Target Verify bucket、Plan Stream 的非对称 overlap、Target NPU Multi-Stream 与 DeepEP recv-hook 出发，建立 Ascend 910C speculative round 的 critical-path 分析框架。
- [DeepSeek-V4 Speculative Decoding 源码解析：Accept Decision 如何选择最终路径？从 Rejection Sampling 到 Parent Tree Recovery](deepseek-v4-accept-decision-source-analysis.md)：区分 Greedy Tree Match、Target-only Tree Sampling 与 classic p/q rejection，解释 accept_index、terminal bonus 和 DSpark correct_len 如何把 speculative future 收敛成唯一历史。
- [DeepSeek-V4 Speculative Decoding 源码解析：Draft Tree 如何落到 KV Cache，Accept 后缓存如何 Commit、Compact 与 Rollback？](deepseek-v4-speculative-kv-commit-compact-rollback-source-analysis.md)：解释 accepted path 如何从 verify-row space 收敛到 committed KV，核清 bonus、overshoot reclaim，以及 Ascend C4/C128 cleanup 的真实边界。
- [DeepSeek-V4 Speculative Decoding 源码解析：多卡场景下 Draft、Verify 与 Accept 如何保持一致](deepseek-v4-speculative-distributed-consistency-source-analysis.md)：解释 speculative control state 在 TP/DP/EP 多卡下如何收敛，比较 DSpark/EAGLE 的同步边界，并区分 HCCL、ZBAL 与 DeepEP。
- [DeepSeek-V4 Prefill / Decode Disaggregation 源码解析：为什么生产系统需要拆分 Prefill 和 Decode](deepseek-v4-pd-disaggregation-source-analysis.md)：追踪 Prefill/Decode 两侧握手、目标 KV 预分配、handoff token、PREBUILT metadata reconstruction，以及 Ascend MemFabric 数据面。

[返回仓库首页](../../README.md)
