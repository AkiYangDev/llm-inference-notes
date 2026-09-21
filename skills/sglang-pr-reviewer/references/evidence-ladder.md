# Evidence Ladder

## 先冻结证据

每个 PR 至少记录：repository、PR number、merged_at、merge commit、base/head、changed files、关键 diff、相关 tests/benchmark。实现解释默认锚定 merge commit，不用后续 main 悄悄替换。

## 四层证据

| 层级 | 含义 | 文章怎么写 |
| --- | --- | --- |
| Source-confirmed | 目标 commit 的源码/测试直接证明 | 可以作为实现事实，保留条件 |
| PR-confirmed | PR body、review、作者说明 | 归因给 PR，不冒充代码直接证明 |
| Measured | CI、benchmark、Profiler、accuracy | 保留硬件、模型、配置、口径和样本范围 |
| Inference / Transfer | 根据前述证据推导的方法论 | 明确是工程推断，不写成源码事实 |

## 冲突处理

对于实现行为：目标 merge commit 的可达源码 > 对应 tests > PR body/review > docs/issues > 第三方解释。PR 动机和设计意图不能仅凭代码猜；若作者说明与代码行为冲突，分别写清。

## Benchmark 规则

- 不只抄提升百分比；记录 baseline/candidate、硬件、模型、并行配置、输入输出长度、并发和测量指标。
- 一次模型/硬件的 threshold 不泛化成全平台规则。
- 没有 benchmark 时明确“未测”，不要用理论收益补成实测。

## 跨平台规则

CUDA / ROCm / Ascend / XPU 分支必须分别核验。一个 NPU label 不代表所有修改都位于 NPU 执行路径；一个 CUDA benchmark 也不能直接证明 Ascend 收益。
