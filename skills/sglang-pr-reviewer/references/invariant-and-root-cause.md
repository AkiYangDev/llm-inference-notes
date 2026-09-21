# Invariant 与 Root Cause

不要把“出错的那一行”自动当成 Root Cause。PR 精读至少区分四层：

1. **Symptom**：用户/CI/Profiler 看到了什么异常；
2. **Proximal cause**：直接导致异常的判断、状态或数据；
3. **Root cause**：为什么系统允许这条错误路径出现；
4. **Invariant**：系统本应始终保持、但被破坏的规则。

## 常见 invariant 家族

| 家族 | 典型问题 |
| --- | --- |
| Single Source of Truth | 配置值、派生值、日志/运行时语义漂移 |
| Ownership | 谁拥有 Tensor、KV、buffer、backend、binding |
| Lifecycle | allocate → use → publish/commit → reclaim 是否闭合 |
| Terminal State | zero-work / error / timeout 是否也能结束状态机 |
| Capacity | write/pack/gather 是否永远不超过真实 buffer / budget |
| Ordering | publish、commit、rollback、free 的先后是否可被打乱 |
| Shape / Contract | Graph replay、Kernel、ModelRunner 的输入约束是否一致 |
| Synchronization | host/device、stream、rank 是否读取了尚未 ready 的状态 |
| Idempotence | retry、replay、重复 publish 是否会二次修改状态 |
| Topology | rank/group/world size 的派生是否来自同一配置语义 |

## 找 invariant 的问题

- 如果只修这一个 if，其他调用点会不会再次出错？
- 哪个值/状态应该只有一个权威来源？
- 这个请求即使没有工作量，是否仍必须结束？
- 哪个 buffer / Tensor 的容量、生命周期或 owner 被默认错了？
- Capture 和 Replay 是否满足同一 contract？
- 哪个状态在 Draft/Target、Prefill/Decode、不同 rank 之间被错误共享？

一个好的 invariant 比具体数字更稳定。例如“AttnTP=8”只是实例，而“所有 Attention width 消费者共享同一派生语义”才是可迁移的 invariant。
