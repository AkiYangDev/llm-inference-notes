# LLM Inference Notes

大模型推理工程文档：原理、源码、部署与性能分析。

Engineering notes on LLM inference, with an emphasis on SGLang and Ascend NPU.

[在线阅读](https://akiyangdev.github.io/llm-inference-notes/)

## 内容导航

| 主题 | 内容范围 |
| --- | --- |
| [推理基础](docs/fundamentals/README.md) | Tensor、Attention、Prefill / Decode、KV Cache |
| [SGLang 源码](docs/sglang/README.md) | 请求链路、Scheduler、批次组织与模型执行 |
| [Ascend 部署](docs/ascend/README.md) | 环境配置、模型部署、算子后端与排障 |
| [分布式推理](docs/distributed/README.md) | TP / DP / EP / PP、通信与数据归属 |
| [投机推理](docs/speculative-decoding/README.md) | Draft / Verify、接受逻辑与缓存状态 |
| [性能分析](docs/performance/README.md) | TTFT、TPOT、吞吐、显存与 Profiling |

## 专题文章

- [SGLang 推理执行链路解析：一次 Chat Completion 请求是如何跑到 Ascend NPU 上的](docs/sglang/sglang-ascend-request-lifecycle.md)：以 DeepSeek-V4 为例，连接请求调度、共享与压缩缓存、Ascend 算子及流式输出，并说明 DSPARK 的验证边界。
- [SGLang Scheduler 源码解析：一个请求是如何被组批、调度并送进 ModelRunner 的](docs/sglang/sglang-scheduler-source-analysis.md)：从 `Req`、`ScheduleBatch`、`ForwardBatch` 三个生命周期切入，追踪 admission、Prefix Cache、Prefill / Decode 与 ModelRunner 边界。
- [SGLang ModelRunner 源码解析：ForwardBatch 如何驱动 DeepSeek 完成一次 Prefill / Decode](docs/sglang/sglang-modelrunner-source-analysis.md)：沿 `ForwardBatch → ModelRunner → DeepSeek-V4 → LogitsProcessor → Sampling` 追踪一次 Prefill / Decode 的真实执行边界。
- [DeepSeek-V4 Attention 源码解析：一个 ForwardBatch 如何变成 Query，并通过 KV Cache 找回历史上下文](docs/sglang/deepseek-v4-attention-source-analysis.md)：追踪 Query 构造、NPU cache write、request-to-page 映射，以及按层选择的 SWA / C4 / C128 历史读取路径。
- [DeepSeek-V4 MoE 推理源码解析：一个 Token 如何经过 Router、All-to-All 和 Expert Parallel 完成一次前向计算](docs/sglang/deepseek-v4-moe-source-analysis.md)：从 Router / HashTopK 追到 Expert ownership、EP dispatch、Expert compute、combine，并解释 Ascend FuseEP 的融合边界。
- [DeepSeek-V4 分布式推理源码解析：TP、EP、DP 三种并行如何共同驱动一次前向计算](docs/sglang/deepseek-v4-distributed-parallel-source-analysis.md)：解释 `dp_size → attn_dp/attn_tp`、Attention→MoE 布局桥、A2A 下 EP=TP，以及 TP32/DP16 在 DSpark 路径中的显式限制。
- [DeepSeek-V4 DSpark 源码解析：Draft Model 为什么需要独立处理 TP/DP/EP Layout？](docs/sglang/deepseek-v4-dspark-parallel-layout-analysis.md)：区分 Dense/MoE draft 的并行上下文，解释 DP/MoE token-count metadata、SpecTpSync、Verify layout，并把 TP32/DP16 的已知事实与待验证根因分开。
- [DeepSeek-V4 Speculative Decoding 源码解析：Draft → Verify → Accept/Reject 如何驱动一次投机推理](docs/sglang/deepseek-v4-speculative-decoding-source-analysis.md)：沿 DSparkWorkerV2 追踪 Draft Block、Verify Planner、Target Verify、greedy/sampling accept、commit_lens 与 Target hidden → Draft state 提交。
- [DeepSeek-V4 Prefill / Decode Disaggregation 源码解析：为什么生产系统需要拆分 Prefill 和 Decode](docs/sglang/deepseek-v4-pd-disaggregation-source-analysis.md)：沿 Prefill bootstrap、Decode prealloc、KV/state transfer、PREBUILT handoff 与首轮 Decode 追踪一次 P→D ownership 迁移，并拆清 Ascend DSV4 的 C4/SWA/C128 wire layout。

## 配套内容

- [代码示例](examples/README.md)：文章配套的可运行示例。
- [图表资源](assets/README.md)：架构图、调用链图和实验图表。
- [写作模板](templates/README.md)：源码解析与部署实践模板。
- [技术写作 Skill](skills/README.md)：源码取证、连续工程叙事、图解与文章评审规则。
- [贡献与维护](CONTRIBUTING.md)：命名、证据要求与检查方式。

文章按具体版本解释实现；原理示例、源码确认与实测结果分别标注。
