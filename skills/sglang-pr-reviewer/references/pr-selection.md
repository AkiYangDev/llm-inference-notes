# PR 选题：什么值得精读？

PR 精读不是 changelog。优先选择能暴露真实系统机制、并能把 patch 提炼成可迁移知识的改动。

## 选题评分

| 维度 | 权重 | 核心问题 |
| --- | ---: | --- |
| 问题清晰度 | 15 | Expected vs Actual 是否能被明确描述？ |
| Invariant 价值 | 20 | 是否能指出一条被破坏的系统不变量？ |
| 可迁移知识 | 20 | 能否形成其他模块也可用的 Bug Pattern / Review Rule？ |
| Diff 可读性 | 10 | 关键修改是否能在有限文件和上下文内还原？ |
| 证据质量 | 15 | 是否有 tests、benchmark、Profiler、review 或明确源码证据？ |
| 系统代表性 | 10 | 是否触及 Scheduler、KV、Spec、Distributed、Kernel、Memory 等核心机制？ |
| 与已有内容互补 | 10 | 是否能补充已有专题，而不是重复完整教程？ |

建议：70 分以上适合正式精读；55～69 分可做短笔记或候选池；低于 55 分通常跳过。分数只用于筛选，不代表 PR 工程质量。

## 优先 PR

- bug fix with a clear invariant；
- performance PR with reproducible measurement；
- memory PR with allocation / lifetime math；
- distributed PR with rank/group/ownership change；
- scheduler / async PR with terminal-state or ordering issue；
- graph / kernel PR with shape、contract 或 fallback 边界。

## 通常不优先

纯 rename、formatting、机械依赖升级、无行为差异的模型名单更新。CI PR 只有在 predicate、coverage、false positive/negative 等机制具有迁移价值时才值得精读。

## 难度标签

- ★☆☆☆☆：1～2 个核心概念，失败路径短，适合第一次读 PR；
- ★★☆☆☆：需要理解一个子系统，如 pinned memory、cache extension；
- ★★★☆☆：涉及状态机、Scheduler、KV 生命周期或通信；
- ★★★★☆：跨多个子系统、Graph、Spec、MoE/DeepEP；
- ★★★★★：需要大量上下游源码、Profiler 或多平台差异才能完整判断。
