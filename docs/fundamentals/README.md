# 推理基础

这里先建立一张从模型语义、推理生命周期、Runtime、内存、分布式、投机推理到 Ascend 910C 执行层的知识地图，再进入各专题源码分析。

## 入门与总导航

- [AI 推理基础设施工作名词表：SGLang、DeepSeek 与 Ascend 910C 从 Token 到 NPU Kernel](ai-infra-working-glossary.md)：串起 Token、KV Cache、Scheduler、TP / DP / EP、DeepEP、DSpark、ModelSlim、CANN、Tiling 与 NPU Kernel，并提供当前 SGLang 源码入口索引和专题文章阅读路线。

## 建议阅读顺序

1. 先读上面的工作名词表，建立“Model → Runtime → Memory → Distributed → Communication → Operator → Kernel → Hardware”的整体地图。
2. 再进入 [SGLang 源码](../sglang/README.md)，沿请求执行链路、Scheduler、ModelRunner、Attention、MoE 与分布式布局逐层下钻。
3. 遇到具体部署、并行、投机推理或性能问题，再回到对应专题文章，而不是孤立地背缩写。

[返回仓库首页](../../README.md)
