# PR History Chain

单个 diff 经常只是一个更长设计演进中的一环。History Chain 用来回答：这个 contract 从哪里来、为什么现在才暴露、当前修复是否依赖其他 PR、后来是否还有 follow-up。

## 最低检查顺序

```text
Issue / user report
        ↓
introducing PR / original abstraction
        ↓
stacked-on / dependency PRs
        ↓
current PR
        ↓
follow-up / regression / cleanup
```

并非每个 PR 都有完整链路。找不到时写 unknown，不为了叙事编造 introducing PR。

## 需要记录的关系

- fixes / closes issue；
- introduced by / regression since；
- stacked on / depends on；
- supersedes / replaces；
- follow-up to；
- shared helper / abstraction originally added by；
- review request 中明确要求后续补的缺口。

## 为什么它影响 Root Cause

例如一个 capacity Bug 可能不是“某函数忘了 if”，而是新 buffer abstraction 只把 bound 接进 compute path，却漏掉 cached-prefix transfer path。知道 introducing PR 后，invariant 会从局部 if 提升为“所有消费者必须共享同一 capacity contract”。

## 版本纪律

- 历史 PR 只用于解释因果，不用后来的代码替代当前 merge commit。
- follow-up 不能倒推出当前 PR 当时已经具备后来能力。
- stacked PR 必须说明基线，否则 GitHub diff 可能包含 parent 改动，导致误判 changed files。
- 若 PR body 引用 benchmark from earlier revision，标明 historical evidence，不假装 final revision 已重跑。
