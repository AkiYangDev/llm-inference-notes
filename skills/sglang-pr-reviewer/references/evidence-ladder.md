# Evidence Ladder

## 先冻结证据

每个 PR 至少记录：repository、PR number、merged_at、merge commit、base/head、changed files、关键 diff、相关 tests/benchmark。实现解释默认锚定 merge commit，不用后续 main 悄悄替换。

## 四类主证据

| 类型 | 含义 | 文章怎么写 |
| --- | --- | --- |
| Source-confirmed | 目标 commit 的源码/测试直接证明 | 可以作为实现事实，保留条件 |
| PR-confirmed | PR body、review、作者说明 | 归因给 PR，不冒充代码直接证明 |
| Measured | CI、benchmark、Profiler、accuracy | 必须标明测量来源与覆盖面 |
| Inference / Transfer | 根据前述证据推导的方法论 | 明确是工程推断，不写成源码事实 |

## Measured 还要标“谁测的”

- **Author-reported**：PR 作者报告的本地或实验环境结果。
- **CI-observed**：可定位 workflow/job 的运行结果；仍需检查测试是否实际执行。
- **Independent reproduction**：Reviewer 在固定版本和记录环境下独立复现。

不要把 Author-reported 自动写成 “we reproduced”。独立复现也不能自动泛化到其他平台/shape。

## 性能证据还要标“测到哪一层”

| Level | 证据 | 能证明什么 |
| --- | --- | --- |
| A | Mechanism | 少 launch/copy/sync、缩小 Tensor、复杂度变化；不等于已加速 |
| B | Microbenchmark | isolated kernel/helper 局部 timing |
| C | Workload | model step / prefill / decode / request 在指定 workload 下收益 |
| D | Serving | TTFT / TPOT / throughput / latency / concurrency 等用户层指标 |

不得用 Level B 的 6× 写成 serving 6×。如果只有 Mechanism，就写“理论上减少某成本 / 源码上移除某同步”，不要发明 timing。

## 冲突处理

对于实现行为：目标 merge commit 的可达源码 > 对应 tests > PR body/review > docs/issues > 第三方解释。PR 动机和设计意图不能仅凭代码猜；若作者说明与代码行为冲突，分别写清。

## Benchmark 规则

- 不只抄提升百分比；记录 baseline/candidate、硬件、模型、并行配置、输入输出长度、并发和测量指标。
- 一次模型/硬件的 threshold 不泛化成全平台规则。
- 历史 revision 的 benchmark 标成 historical evidence；final revision 未重跑就不能写成 final measured result。
- 没有 benchmark 时明确“未测”，不要用理论收益补成实测。

## 跨平台规则

CUDA / ROCm / Ascend / XPU 分支必须分别核验。一个 NPU label 不代表所有修改都位于 NPU 执行路径；一个 CUDA benchmark 也不能直接证明 Ascend 收益。
