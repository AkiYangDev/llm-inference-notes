# SGLang 源码

请求入口、调度、批次数据、模型执行和输出处理。

> 第一次接触 SGLang 时，建议先读 [AI 推理基础设施工作名词表](../fundamentals/ai-infra-working-glossary.md) 建立整体坐标，再读 [SGLang Decode 源码入门篇](../fundamentals/sglang-decode-source-reading-primer.md)，沿一枚 Token 走通 ForwardBatch、KV Cache、Attention、MoE、Logits 与 Sampling，最后再进入下面的分专题源码。

## 入门桥梁

- [从一次 Decode 读懂 SGLang：Tensor、缓存与下一枚 Token](../fundamentals/sglang-decode-source-reading-primer.md)：严格核清普通 Decode 的 L→L+1 时间语义、V4 mHC 的 [T,4,H]↔[T,H] 边界、Ascend 910C DSV4 的 SWA/C4/C128 ownership，以及 DP Attention / Vocab Parallel 下 Logits 与 Sampling 的 Group 边界。

## 核心执行链

- [SGLang 请求生命周期：从 Chat Completion 到 Ascend NPU](sglang-ascend-request-lifecycle.md)：DeepSeek-V4、`dsv4` Ascend 后端与 DSpark 生成边界。
- [SGLang Scheduler：请求怎样组批、执行和持续生成](sglang-scheduler-source-analysis.md)：从 `Req` 到 `ScheduleBatch`、`ForwardBatch`，解释 admission、Prefix Cache、Prefill / Decode 调度与 ModelRunner 边界。
- [SGLang ModelRunner：九个输入 Token 怎样产生两行 Logits](sglang-modelrunner-source-analysis.md)：解释 `ForwardBatch` 如何经过 Graph / Eager 分发进入 DeepSeek-V4，并从 packed hidden states 走到 logits 与 Sampling。
- [DeepSeek-V4 Attention：Query、缓存写入与历史寻址](deepseek-v4-attention-source-analysis.md)：从 `MQALayer` 追到 Query、SWA/C4/C128 缓存、page table 与 Ascend DSV4 Attention backend。
- [DeepSeek-V4 MoE：Token 从路由到专家计算，再回到原位置](deepseek-v4-moe-source-analysis.md)：解释 V4 Router / HashTopK、shared experts、EP token routing、DeepEP 与 Ascend FuseEP 的 runtime 边界。

## 评测与故障排查

- [GSM8K 精度排查：从 200 题失败到全量与逐题对照](gsm8k-accuracy-debugging-from-200-to-full-evaluation.md)：从固定 200 题的失败、DeepEP 配置对照与逐题输出，追到 1314 题全量评测，区分门禁覆盖、输出稳定性和代码回退。

## 相关专题

- [分布式推理](../distributed/README.md)：TP / DP / EP、Rank / Group、Collective、DSpark Parallel Layout 与多卡一致性。
- [投机解码](../speculative-decoding/README.md)：Draft / Verify / Accept、MTP、EAGLE、Tree Attention、KV Commit 与自适应 Verify。
- [性能分析](../performance/README.md)：Speculative Decoding break-even、Ascend 910C Critical Path 与 Profiling。
- [Prefill / Decode Disaggregation](deepseek-v4-pd-disaggregation-source-analysis.md)：P→D ownership、KV/state transfer 与 Ascend MemFabric。

[返回仓库首页](../../README.md)
