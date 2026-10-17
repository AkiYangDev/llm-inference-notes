# Ascend 部署

运行环境、模型部署、量化执行、设备后端与故障排查。对外统一使用 Ascend 910C 作为主要硬件称呼。

## 算子与量化

- [DeepSeek-V4 W8A8 执行链：从量化权重到 Ascend INT8 算子](deepseek-v4-w8a8-ascend-910c.md)：从官方 ModelSlim `W8A8_DYNAMIC` 配方出发，追踪 Dense / MoE 的 INT8 数据流、DeepEP INT8 dispatch、`npu_quant_matmul` 到 op-plugin / ACLNN / CANN QuantBatchMatmulV3 的真实边界。
- [DeepSeek-V4 W8A8 性能分析：小 M、NZ 布局与测量方法](../performance/deepseek-v4-w8a8-ascend-910c-performance.md)：承接 W8A8 执行链，进一步核对 DynamicQuant 小 M 并行度、WeightNz Tiling、NZ/ND A/B 与 MoE Expert-local `M_e` 性能边界。
- [SGLang 请求生命周期：从 Chat Completion 到 Ascend NPU](../sglang/sglang-ascend-request-lifecycle.md)：以 DeepSeek-V4 为例，连接请求调度、ForwardBatch / ModelRunner、Ascend 算子入口与流式输出。

[返回仓库首页](../../README.md)
