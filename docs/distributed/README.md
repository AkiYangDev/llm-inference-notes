# 分布式推理

TP、DP、EP、PP 与数据归属、通信正确性。

## 入门与坐标系

- [模型为什么必须多卡？从一张 Ascend 910C 放不下 DeepSeek-V4 到 TP / DP / EP](why-large-models-need-multi-card-tp-dp-ep.md)：先从 304B 总参数、13B Active Params 和本地 HBM 约束说明“为什么必须分布式”，再区分 TP、普通 DP / DP Attention 与 EP 分别切什么，并用 TP16 / DP8 / EP16 建立进入 Rank / Group / Collective 源码前的整体地图。
- [SGLang 里的 Rank 和 Group 到底是什么？用 DeepSeek-V4 画清 TP Rank、DP Rank、EP Rank](sglang-rank-group-deepseek-v4.md)：从 WORLD / TP Group 入手，区分 Native DP 与 DPA 下的 `dp_rank`，画清 Attention 的 `DP × CP × TP`、MoE 的 `DP × EP × TP`，并解释 Ascend 910C 上独立 `_MOE_EP` Group 的边界。
- [SGLang 里的 AllReduce、AllGather、ReduceScatter、All-to-All 到底在搬什么？用 DeepSeek-V4 画清 Group、Tensor 和通信方向](sglang-collectives-deepseek-v4.md)：承接 Rank / Group，解释 TP Linear、DPA gather、MoE ReduceScatter，以及 Ascend DeepEP / FuseEP 的真实数据所有权变化与通信边界。
- [DeepSeek-V4 分布式推理源码解析：TP、EP、DP 三种并行如何共同驱动一次前向计算](../sglang/deepseek-v4-distributed-parallel-source-analysis.md)：从 Rank / Group 坐标进入真实 Attention→MoE layout bridge、collective 与 Ascend 通信边界。

## DSpark 与投机解码的分布式边界

- [DeepSeek-V4 DSpark 源码解析：Draft Model 为什么需要独立处理 TP/DP/EP Layout？](deepseek-v4-dspark-parallel-layout-analysis.md)：理解 Draft / Target / Verify 使用不同并行布局时的同步与 ownership 问题。
- [DeepSeek-V4 Speculative Decoding 源码解析：多卡场景下 Draft、Verify 与 Accept 如何保持一致](deepseek-v4-speculative-distributed-consistency-source-analysis.md)：拆分 speculative control、logical layout 与 model-data collective，比较 DSpark SpecTpSync 与 EAGLE Accept broadcast。

[返回仓库首页](../../README.md)
