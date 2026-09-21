# Review Questions

Code Review Rule 是结论；Review Questions 是用来发现新问题的探针。优先把规则翻译成能在陌生代码上提问的问题。

## 问题族

### Semantic / Derived State
- 这里读的是 configured value，还是已经派生后的 runtime state？
- 是否存在另一个 helper / coordinator 已经定义同一语义？
- 日志、metrics、runtime 和 tests 是否消费同一个来源？

### Ownership / Lifetime
- 这个对象是 owner、view、alias 还是 borrowed handle？
- 它进入长生命周期容器后会额外保活什么资源？
- eviction / free / rollback 真正释放的是账面对象还是 backing resource？

### State Machine / Async
- zero-work、timeout、error、cancel 是否都能进入 terminal state？
- publish、commit、rollback、free 的顺序能否被异步路径打乱？
- callback / future / queue 中是否存在“没人再推进”的状态？

### Capacity / Bounds
- 计算 chunk、transfer size、pack capacity、workspace 上界来自同一维度吗？
- 有无路径能绕开 bound 或在 resize 后继续使用旧容量？

### Performance
- 这是少一次 op / copy / sync 的机制事实，还是已经有 timing？
- benchmark 测的是 kernel、model step、request 还是 serving？
- 阈值是否只对某个 shape / device 成立？

### Validation
- happy path 测了，failure path / zero-size / boundary / fallback 测了吗？
- 哪个平台、dtype、shape、并发没有覆盖？
- test green 是真的执行了目标测试，还是 gate 跳过了？

### History / Compatibility
- 这个 contract 是哪个 PR 引入的？当前 fix 是补漏还是改变设计？
- 是否 stacked on 其他 PR？若缺 parent，当前 diff 会不会被误读？
- follow-up 是否说明当前 PR 仍留有已知缺口？

## 输出方式

正式文章不需要把所有问题机械列出来。至少保留 2～5 个最能迁移到下一次 Code Review 的问题；候选 PR 学习笔记可以直接输出 Review Questions 清单。
