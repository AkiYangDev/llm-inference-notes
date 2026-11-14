# 分布式推理

[EPLB 专题：从专家热度、权重迁移到服务收益](eplb-study-guide.md)，按问题查阅，也可顺着系列阅读。

TP、DP、EP、PP 与数据归属、通信正确性。

## 入门与坐标系

- [模型怎样跨多卡运行：从显存容量到 TP、DP Attention 与 EP](why-large-models-need-multi-card-tp-dp-ep.md)：先从 304B 总参数、13B Active Params 和本地 HBM 约束说明“为什么必须分布式”，再区分 TP、普通 DP / DP Attention 与 EP 分别切什么，并用 TP16 / DP8 / EP16 建立进入 Rank / Group / Collective 源码前的整体地图。
- [SGLang Rank 与 Group：读懂 TP、DP Attention 和 EP 坐标](sglang-rank-group-deepseek-v4.md)：从 WORLD / TP Group 入手，区分 Native DP 与 DPA 下的 `dp_rank`，画清 Attention 的 `DP × CP × TP`、MoE 的 `DP × EP × TP`，并解释 Ascend 910C 上独立 `_MOE_EP` Group 的边界。
- [SGLang Collective：从张量分片到 Attention 与 MoE 通信](sglang-collectives-deepseek-v4.md)：承接 Rank / Group，解释 TP Linear、DPA gather、MoE ReduceScatter，以及 Ascend DeepEP / FuseEP 的真实数据所有权变化与通信边界。
- [DeepSeek-V4 分布式 Forward：Attention 与 MoE 怎样交换布局](../sglang/deepseek-v4-distributed-parallel-source-analysis.md)：从 Rank / Group 坐标进入真实 Attention→MoE layout bridge、collective 与 Ascend 通信边界。

## DSpark 与投机解码的分布式边界

- [DeepSeek-V4 DSpark 的并行限制与 Draft 布局](deepseek-v4-dspark-parallel-layout-analysis.md)：理解 Draft / Target / Verify 使用不同并行布局时的同步与 ownership 问题。
- [多卡投机解码：Draft、Verify 与 Accept 怎样保持一致](deepseek-v4-speculative-distributed-consistency-source-analysis.md)：拆分 speculative control、logical layout 与 model-data collective，比较 DSpark SpecTpSync 与 EAGLE Accept broadcast。

## 专家布局与运行时迁移

- [SGLang 的动态 EPLB：GPU 和 NPU 到底共用了什么，又差在哪里？](sglang-eplb-gpu-npu-implementation.md)：用四张源码图对照计数入口、公共 planner、权重表示、共享 updater 与通信组，解释 NPU 适配应放在哪一层。

- [EPLB 到底改了什么？从一次路由到一次专家搬家](eplb-routing-to-migration-primer.md)：用四位逻辑专家、六个物理槽位，串起路由、副本、统计、规划、权重迁移与映射提交。
- [热度表到底记录了什么？把 NPZ 里的专家计数读明白](eplb-count-npz-reading.md)：实际读取两份原始 NPZ，解释记录、层、物理槽位、rank 归约、角色与布局映射，并提供完整脚本与数据。
- [同样的新布局，怎样少搬一点专家权重？](eplb-minimize-weight-movement.md)：区分原槽位保留、本地重排和远端引入，对照现有 updater 的复用与源保护，再看稳定槽位的收益范围。

- [到了更新间隔，为什么不一定要搬专家？](eplb-skip-unnecessary-migration.md)：从均衡度阈值到整层及本卡筛选，解释更新流程的退出条件。
- [16 个冗余专家怎么分：从专家热度到副本，再到物理布局](eplb-redundant-expert-allocation.md)：用完整小例子走过逐次副本分配、固定槽位 packing 和映射生成，再核对真实 16 个额外位置的分配。

- [静态 EPLB 之后：动态专家负载均衡怎样安全地迁移权重？](ascend-dynamic-eplb-weight-migration.md)：沿统计、规划、ND staging、权重回填和metadata提交，理解动态EPLB的正确性与性能边界。

[返回仓库首页](../../README.md)
