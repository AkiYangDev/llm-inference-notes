# Impact Surface

Root Cause 回答“为什么错”，Impact Surface 回答“这个错误在什么条件下以什么方式伤害系统”。不要只写一句“会 OOM / 会变慢 / 会报错”。

## 六个问题

| 维度 | 要回答的问题 |
| --- | --- |
| Trigger | 什么输入、配置、时序或资源状态会触发？ |
| Symptom | 用户、日志、CI、Profiler 或 Runtime 看到什么？ |
| Affected Config | 哪些模型、平台、并行模式、shape、并发受影响？ |
| User-visible Impact | 请求失败、错误输出、延迟、吞吐、内存、CI 成本分别有什么变化？ |
| Blast Radius | 单 Tensor、单请求、单 Worker、整机、整个 CI / serving 集群？ |
| Recovery / Persistence | 自恢复、重试可恢复、必须重启，还是长期累积？ |

## 写作规则

- Impact 必须和已知证据一致；没有 production 事故数据时不要编造严重性。
- “可能影响”与“已复现影响”分开写。
- 资源类 Bug 要区分 peak、retained、leaked、pinned、fragmented 等不同现象。
- 正确性 Bug 要区分 silent wrong result、显式 error、仅 observability wrong。
- 性能 Bug 要区分局部 kernel、step、request 和 serving 层影响。

## 示例：PR #39120

Trigger：批量多模态 encoder 返回一个整体 Tensor，`torch.split` 后的小 View 被放入长期 Cache。Symptom：Cache 账面按 payload 计费，但历史 Batch Storage 被 View 继续保活。Blast Radius 不是“一个 View 占 N 倍独立内存”，而是不同历史 Batch 可能分别被少量 View pin 住。更准确的词是 unintended retention，不是失去引用的永久 leak。
