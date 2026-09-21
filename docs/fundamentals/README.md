# 推理基础

这里先建立从模型语义、推理生命周期、Runtime、内存、分布式、投机解码到 Ascend 910C 执行层的基础直觉，再进入各专题源码分析。

## 入门与总导航

- [AI 推理基础设施工作名词表：SGLang、DeepSeek 与 Ascend 910C 从 Token 到 NPU Kernel](ai-infra-working-glossary.md)：建立 Token、KV Cache、Scheduler、TP / DP / EP、DeepEP、DSpark、ModelSlim、CANN、Tiling 与 NPU Kernel 的全局坐标。
- [为什么 Prefill 和 Decode 明明跑的是同一个模型，性能却完全不同？](prefill-vs-decode-performance.md)：从真实 Dense / MoE shape、Weight reuse、DSV4 SWA/C4/C128 历史读取与 Ascend 910C Profiling 指标建立第一套性能直觉。

## 建议阅读顺序

1. 先读工作名词表，建立“Model → Runtime → Memory → Distributed → Communication → Operator → Kernel → Hardware”的整体地图。
2. 再读 Prefill / Decode 性能文章，理解为什么同一套模型权重会形成两种完全不同的 Workload Geometry，以及 KV Cache、Continuous Batching、Chunked Prefill、投机解码和 PD 分离为什么会出现。
3. 然后进入 [SGLang 源码](../sglang/README.md)，沿请求执行链路、Scheduler、ModelRunner、Attention 与 MoE 逐层下钻。
4. 遇到量化、并行、投机解码或性能诊断问题，再进入对应专题，而不是孤立地背缩写。

[返回仓库首页](../../README.md)
