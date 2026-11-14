# EPLB 专题：从专家热度、权重迁移到服务收益

EPLB 要解决的问题很具体：同一层 MoE 的工作没有均匀落到设备上，部分 rank 忙，其他 rank 等。增加热点专家的副本、重新安排位置，可以改善这种分配；但它同时引入统计、规划、搬权重和切换映射的成本。

这个专题围绕 SGLang 与 Ascend 的实际实现展开。先用小例子建立逻辑专家、物理副本与槽位的关系，再读统计和迁移源码，最后用已有 profiling、显存、服务性能与正确性材料判断优化值不值得保留。

如果是第一次接触 EPLB，先读下面的三篇入门。已经在排查问题，可以直接使用按问题索引。每篇的固定版本和实测条件写在正文；教学数字、代码事实与设备结果分别保留来源。

## 先读这三篇

1. [EPLB 到底改了什么？从一次路由到一次专家搬家](eplb-routing-to-migration-primer.md)：先分清模型要执行谁、权重放哪里，以及什么情况下才搬权重。
2. [16 个冗余专家怎么分](eplb-redundant-expert-allocation.md)：用四专家、两冗余走完副本分配与槽位安排，再对应真实布局。
3. [热度表到底记录了什么？把 NPZ 里的专家计数读明白](eplb-count-npz-reading.md)：实际打开两份 200 条记录，读懂维度、归约、角色与映射。

想先看整项工程怎么演进，可以从[一个 EPLB PR 怎样收敛](../performance/eplb-pr49701-commit-evolution.md)进入。它按 PR #49701 的实际 commit 解释核心改动、兼容性修复和撤回的尝试，再连接到各篇深挖。

## 一、从统计开始：窗口里究竟装了什么

统计的单位不是客户端请求。一次 forward 可以处理多个请求或多个候选，一个请求也可以经历很多次 forward；物理槽位的路由计数，还要按正确布局汇总成逻辑专家负载。

| 你在问什么 | 对应文章 |
|---|---|
| NPZ 的三个轴和每个字段是什么 | [把专家计数读明白](eplb-count-npz-reading.md) |
| 收到的路由为什么不全是有效工作 | [DeepEP 收到多少，与模型真正执行多少](../performance/deepep-received-vs-valid-routes.md) |
| 加一条零记录，为什么求和没变而决策变了 | [零记录与滚动窗口](../performance/eplb-zero-record-rolling-window.md) |
| 数据已经采到，为什么导出还能触发故障 | [专家计数导出与 HCCL / Gloo](../performance/expert-count-export-hccl-gloo.md) |
| P 和 D 的 I、W 应该怎样理解 | [两端为什么不能照抄同一套参数](../performance/eplb-prefill-decode-parameter-design.md) |
| 新布局为什么会追着旧热点跑 | [统计历史与热点延迟](../performance/dynamic-eplb-stale-hotspots.md) |

零记录文章使用历史缺陷的固定版本。它说明怎样验证窗口组成和修复生命周期，不把历史现象当作当前主线状态。新版本排障，应先确认自己部署的 recorder 与 runner 代码。

## 二、从热度到布局：要放几份，又放在哪里

Planner 要同时满足每位逻辑专家的基本副本、冗余预算与各 rank 的槽位容量。统计看起来更均匀，也不保证真实设备时间一定缩短。

- [冗余副本分配与 packing](eplb-redundant-expert-allocation.md)：热度为什么要除以副本数，热点为什么能连续拿到多个位置。
- [到了更新间隔，为什么不一定要搬专家](eplb-skip-unnecessary-migration.md)：检查、均衡度阈值、整层变化和本卡参与，分别在哪里决定继续或退出。
- [同样的新布局，怎样少搬一点权重](eplb-minimize-weight-movement.md)：原槽位保留、本地重排与远端引入分别要做什么，现有 updater 已经复用了哪些数据。
- [相同 token 数，为什么有的卡还是更慢](../performance/moe-equal-token-counts-unequal-latency.md)：把专家分组、尾块、通信到达与等待放回计数指标旁边。

如果业务组成相对固定，也可以先看[场景静态 EPLB](../performance/ascend-910c-prefill-scenario-static-eplb.md)。静态布局与动态更新分别解决不同的适应问题，比较时要对齐实际 KV、冗余与缓存条件。

## 三、从布局到权重：数据、地址和通信怎样一起切换

映射表很小，表指向的专家数据却必须真实到位。这里最容易混淆三件事：格式转换、源数据保护，以及对原模型存储的最终写入。

- [搬专家权重，还需要 NZ→ND 吗](../performance/ascend-eplb-physical-storage-migration.md)：当前物理字节路径的主要入口，包含源代码、原始服务对照与截图。
- [图捕获以后，为什么不能随便换张量](../performance/graph-capture-tensor-storage.md)：原位改内容、替换存储和设备依赖有什么区别。
- [复用 TP 通信组，到底省在哪里](../performance/ascend-eplb-tp-world-memory.md)：分清默认 WORLD、已有 TP 与首次激活的通信资源。
- [后台搬权重，前台就不用等了吗](../performance/async-eplb-foreground-wait.md)：独立缓冲、源读取、就绪协调与前台提交怎样形成依赖。
- [搬完没报错，怎样证明模型仍在算对](../performance/eplb-weight-migration-correctness.md)：字节、scale、逻辑映射、图读取、固定状态前向和任务评测分别查什么。

两篇历史文章也保留在专题中：[动态迁移的旧 ND 路径](ascend-dynamic-eplb-weight-migration.md)与[逐层转换为何仍堆出多层临时张量](../performance/ascend-eplb-memory-bounded-staging.md)。它们解释方案演进，后续物理路径已删除转换中转，不应把两代代码拼成一份当前实现。

## 四、看成本：到底花了多少时间和显存

先数实际工作，再解释耗时。变化槽位数、远端新增专家数、传输字节数、通信 API 时间与完整 update 墙钟，分别对应不同对象。

- [一次更新到底搬了多少数据](../performance/eplb-migration-byte-budget.md)：从真实专家块、辅助参数和本地复用算到每条瓶颈资源的时间下限。
- [一轮更新的时间究竟花在哪里](../performance/eplb-update-time-breakdown.md)：沿 CPU 提交、设备通信、等待与目标写入读 profiling。
- [EPLB 的收益上限在哪](../performance/eplb-speedup-upper-bound.md)：局部变快以后，先算完整 forward 和周期运行能省多少。
- [多放一份专家，还是多留一点 KV](../performance/eplb-expert-kv-memory-budget.md)：常驻副本、统计、scratch 与请求容量怎样一起占用显存。

物理路径对旧 ND 路径的显存与 update 对照，和动态 EPLB 的 OFF/ON 服务对照，是两套实验。前者比较迁移实现，后者比较启用运行时更新；数字不能跨组拼成同一条收益链。

## 五、看收益：先固定一次执行，再验证整个服务

[动态 EPLB 的 Decode 收益怎么测](../performance/dynamic-eplb-decode-benefit-measurement.md)给出两层对照：固定真实输入、batch、上下文与 KV，执行 A→update→B→restore A；随后再用一致负载做持续 OFF/ON，确认吞吐、延迟、失败与投机指标。

[吞吐提高，平均 TPOT 为什么变差](../performance/tpot-worse-throughput-better.md)分析一组真实 PD 服务结果。P 端更快可能让 D 实际 batch 增大，服务产出与单请求生成节奏需要一起解释，不能只选一个方向写结论。

如果输出文字发生变化，接着看[同样输入换个 batch，为什么答案会变](../performance/same-prompt-different-batch-numerics.md)。先定位同一前缀上的第一处分叉，区分数据、索引、映射错误与计算路径差异；temperature=0 和固定 seed 不能替代这套检查。

## 沿 PR 的优化点阅读

| 演进问题 | 主要提交或证据 | 读哪篇 |
|---|---|---|
| NZ→ND 中转与独立写回 | `4c9a1c24` | [物理存储重构](../performance/ascend-eplb-physical-storage-migration.md) |
| 打包 ND 的 dtype view 与 reshape | `83a22098` | [沿提交看兼容性修复](../performance/eplb-pr49701-commit-evolution.md) |
| 视图命名、存储描述与生命周期 | `a9430a31`、`a4d6fc31` | [提交演进](../performance/eplb-pr49701-commit-evolution.md)、[图与地址](../performance/graph-capture-tensor-storage.md) |
| TP 复用、预热和拓扑约束 | `d84edc66`、`240f795f`、`2914be60` | [TP / WORLD 显存实验](../performance/ascend-eplb-tp-world-memory.md) |
| FP4 packed NZ 格式 | `e7fc1cdd`、`a29d6456` | [格式补全与边界保留](../performance/eplb-pr49701-commit-evolution.md) |
| PP 与计划/复制分支的收敛 | `4802d2bb`、`08865d16` | [哪些尝试最后撤回了](../performance/eplb-pr49701-commit-evolution.md) |
| draft 与 padding 的统计边界 | 原始计数与固定源码 | [零记录](../performance/eplb-zero-record-rolling-window.md)、[有效路由](../performance/deepep-received-vs-valid-routes.md) |

[提交索引 JSON](../../assets/eplb-study-lab/commit-index.json)保留完整 SHA；[算例包](https://raw.githubusercontent.com/AkiYangDev/llm-inference-notes/main/assets/eplb-study-lab/examples.zip)汇总负载、槽位复用、I/W 时间尺度、收益上限与字节预算的 CPU 计算。真实 NPZ 放在热度表文章的独立下载包中。

可以按[系列阅读顺序](https://akiyangdev.github.io/llm-inference-notes/series/#eplb)继续阅读并保存本地进度，也可以从这页直接跳到当前遇到的问题。
