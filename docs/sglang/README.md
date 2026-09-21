# SGLang 源码

请求入口、调度、批次数据、模型执行和输出处理。

> 第一次接触 SGLang 时，建议先读 [AI 推理基础设施工作名词表](../fundamentals/ai-infra-working-glossary.md) 建立整体坐标，再读 [SGLang Decode 源码入门篇](../fundamentals/sglang-decode-source-reading-primer.md)，沿一枚 Token 走通 ForwardBatch、KV Cache、Attention、MoE、Logits 与 Sampling，最后再进入下面的分专题源码。

## 入门桥梁

- [第一次读 SGLang 源码，应该先看懂什么？用 DeepSeek-V4 一次 Decode 串起 Tensor、KV Cache、Attention、MoE 与 Sampling](../fundamentals/sglang-decode-source-reading-primer.md)：严格核清普通 Decode 的 L→L+1 时间语义、V4 mHC 的 [T,4,H]↔[T,H] 边界、Ascend 910C DSV4 的 SWA/C4/C128 ownership，以及 DP Attention / Vocab Parallel 下 Logits 与 Sampling 的 Group 边界。

## 核心执行链

- [SGLang 推理执行链路解析：一次 Chat Completion 请求是如何跑到 Ascend NPU 上的](sglang-ascend-request-lifecycle.md)：DeepSeek-V4、`dsv4` Ascend 后端与 DSpark 生成边界。
- [SGLang Scheduler 源码解析：一个请求是如何被组批、调度并送进 ModelRunner 的](sglang-scheduler-source-analysis.md)：从 `Req` 到 `ScheduleBatch`、`ForwardBatch`，解释 admission、Prefix Cache、Prefill / Decode 调度与 ModelRunner 边界。
- [SGLang ModelRunner 源码解析：ForwardBatch 如何驱动 DeepSeek 完成一次 Prefill / Decode](sglang-modelrunner-source-analysis.md)：解释 `ForwardBatch` 如何经过 Graph / Eager 分发进入 DeepSeek-V4，并从 packed hidden states 走到 logits 与 Sampling。
- [DeepSeek-V4 Attention 源码解析：一个 ForwardBatch 如何变成 Query，并通过 KV Cache 找回历史上下文](deepseek-v4-attention-source-analysis.md)：从 `MQALayer` 追到 Query、SWA/C4/C128 缓存、page table 与 Ascend DSV4 Attention backend。
- [DeepSeek-V4 MoE 推理源码解析：一个 Token 如何经过 Router、All-to-All 和 Expert Parallel 完成一次前向计算](deepseek-v4-moe-source-analysis.md)：解释 V4 Router / HashTopK、shared experts、EP token routing、DeepEP 与 Ascend FuseEP 的 runtime 边界。

## 相关专题

- [分布式推理](../distributed/README.md)：TP / DP / EP、Rank / Group、Collective、DSpark Parallel Layout 与多卡一致性。
- [投机解码](../speculative-decoding/README.md)：Draft / Verify / Accept、MTP、EAGLE、Tree Attention、KV Commit 与自适应 Verify。
- [性能分析](../performance/README.md)：Speculative Decoding break-even、Ascend 910C Critical Path 与 Profiling。
- [Prefill / Decode Disaggregation](deepseek-v4-pd-disaggregation-source-analysis.md)：P→D ownership、KV/state transfer 与 Ascend MemFabric。

[返回仓库首页](../../README.md)
