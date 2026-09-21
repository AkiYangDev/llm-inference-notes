# 分布式推理

TP、DP、EP、PP 与数据归属、通信正确性。

## 入门与坐标系

- [SGLang 里的 Rank 和 Group 到底是什么？用 DeepSeek-V4 画清 TP Rank、DP Rank、EP Rank](sglang-rank-group-deepseek-v4.md)：从 WORLD / TP Group 入手，区分 Native DP 与 DPA 下的 `dp_rank`，画清 Attention 的 `DP × CP × TP`、MoE 的 `DP × EP × TP`，并解释 Ascend 910C 上独立 `_MOE_EP` Group 的边界。

## 进阶阅读

- [DeepSeek-V4 分布式推理源码解析：TP、EP、DP 三种并行如何共同驱动一次前向计算](../sglang/deepseek-v4-distributed-parallel-source-analysis.md)：从 Rank / Group 坐标进入真实 Attention→MoE layout bridge、collective 与 Ascend 通信边界。
- [DeepSeek-V4 DSpark 源码解析：Draft Model 为什么需要独立处理 TP/DP/EP Layout？](../sglang/deepseek-v4-dspark-parallel-layout-analysis.md)：继续理解 Draft / Target / Verify 使用不同并行布局时的同步与 ownership 问题。

[返回仓库首页](../../README.md)
