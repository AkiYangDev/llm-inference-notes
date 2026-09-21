# PR 类型 Playbook

固定六章只提供叙事骨架；不同 PR 的分析重点不同。每种类型都先做 Impact Surface，再选最能暴露 Root Cause 的图和 Validation Surface。

| 类型 | 必查对象 | 最有用的图/表 | Review 重点 | 常见过度解读 |
| --- | --- | --- | --- | --- |
| Correctness / Observability | invariant、derived state、错误路径 | Before/After 数据流 | truth source 是否统一 | 把 log-only 写成 runtime fix |
| Memory | ownership、tensor lifetime、workspace、temporary、peak/retained accounting | 生命周期 / HBM 账本 | 账面资源与实际保活资源是否同粒度 | 把 retained 当 leak，或把某次 peak 当通用上界 |
| Performance | critical path、sync、overlap、benchmark methodology | Timeline | performance evidence 属于 A/B/C/D 哪层 | 只凭少 op 就宣称 serving 加速 |
| Distributed | rank、group、ownership、collective、wire layout | 拓扑 / ownership | 哪个 group / buffer contract 被破坏 | 把 TP×DP 简单相乘或跨平台泛化 |
| Scheduler / Async | queue、state transition、timeout、terminal state | 状态机 | zero-work/error/cancel 是否也终止 | 只看 happy path |
| Speculative | Draft、Verify、Accept、Commit、rollback | 状态/时序图 | 多阶段 state contract 与 publish fence | 把 acceptance proxy 当统一数学定义 |
| KV Cache | allocation、page mapping、write、publish、reclaim | 生命周期 / page ownership | logical len / committed KV / physical page 是否一致 | 混淆 logical len 与 committed KV |
| Kernel | shape、dtype、tiling、launch、fallback、capability probe | shape 表 / threshold 图 | unsupported capability 是否正确 fallback | “fusion/INT8 永远更快” |
| Graph | capture/replay contract、pointer、bucket、padding | Capture vs Replay | capture 与 replay 是否共享 contract | 把 graph enable 等同覆盖全部路径 |
| CI / Infra | predicate、path filter、coverage、gating | truth table | green 是否真的执行目标测试 | 把 CI green 当功能正确证明 |

一篇文章可以有多个类型，但只选择真正影响 Root Cause 的主类型，避免把所有 playbook 都塞进正文。
