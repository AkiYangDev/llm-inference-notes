# PR 类型 Playbook

固定六章只提供叙事骨架；不同 PR 的分析重点不同。

| 类型 | 必查对象 | 最有用的图/表 | 常见过度解读 |
| --- | --- | --- | --- |
| Correctness / Observability | invariant、derived state、错误路径 | Before/After 数据流 | 把 log-only 写成 runtime fix |
| Memory | tensor lifetime、workspace、temporary、peak accounting | 生命周期 / HBM 账本 | 把某次 peak 当通用上界 |
| Performance | critical path、sync、overlap、benchmark methodology | Timeline | 只凭理论减少 op 就宣称加速 |
| Distributed | rank、group、ownership、collective、wire layout | 拓扑 / ownership | 把 TP×DP 简单相乘或跨平台泛化 |
| Scheduler / Async | queue、state transition、timeout、terminal state | 状态机 | 只看 happy path |
| Speculative | Draft、Verify、Accept、Commit、rollback | 状态/时序图 | 把 acceptance proxy 当统一数学定义 |
| KV Cache | allocation、page mapping、write、publish、reclaim | 生命周期 / page ownership | 混淆 logical len 与 committed KV |
| Kernel | shape、dtype、tiling、launch、fallback | shape 表 / threshold 图 | “fusion/INT8 永远更快” |
| Graph | capture/replay contract、pointer、bucket、padding | Capture vs Replay | 把 graph enable 等同真实覆盖全部路径 |
| CI / Infra | predicate、path filter、coverage、gating | truth table | 把 CI green 当功能正确证明 |

一篇文章可以有多个类型，但只选择真正影响 Root Cause 的主类型，避免把所有 playbook 都塞进正文。
