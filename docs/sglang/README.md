# SGLang 源码

请求入口、调度、批次数据、模型执行和输出处理。

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
- [DeepSeek-V4 Speculative Decoding 源码解析：EAGLE Draft Tree 如何组织多分支候选，并驱动一次 Verify](deepseek-v4-eagle-draft-tree-source-analysis.md)：追踪 EAGLE 多分支候选的累计 score、Tree topology、Mask、Target traversal、KV compaction，并区分 Runtime Tree 能力与 Ascend dsv4 paged-tree 当前限制。
- [DeepSeek-V4 Prefill / Decode Disaggregation 源码解析：为什么生产系统需要拆分 Prefill 和 Decode](deepseek-v4-pd-disaggregation-source-analysis.md)：追踪 Prefill/Decode 两侧握手、目标 KV 预分配、handoff token、PREBUILT metadata reconstruction，以及 Ascend MemFabric 数据面。

[返回仓库首页](../../README.md)
