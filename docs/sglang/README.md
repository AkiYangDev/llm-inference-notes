# SGLang 源码

请求入口、调度、批次数据、模型执行和输出处理。

## 专题文章

- [SGLang 推理执行链路解析：一次 Chat Completion 请求是如何跑到 Ascend NPU 上的](sglang-ascend-request-lifecycle.md)：DeepSeek-V4、`dsv4` Ascend 后端与 DSPARK 生成边界。
- [SGLang Scheduler 源码解析：一个请求是如何被组批、调度并送进 ModelRunner 的](sglang-scheduler-source-analysis.md)：从 `Req` 到 `ScheduleBatch`、`ForwardBatch`，解释 admission、Prefix Cache、Prefill / Decode 调度与 ModelRunner 边界。
- [SGLang ModelRunner 源码解析：ForwardBatch 如何驱动 DeepSeek 完成一次 Prefill / Decode](sglang-modelrunner-source-analysis.md)：解释 `ForwardBatch` 如何经过 Graph / Eager 分发进入 DeepSeek-V4，并从 packed hidden states 走到 logits 与 Sampling。

[返回仓库首页](../../README.md)
