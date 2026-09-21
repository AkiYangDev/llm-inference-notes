# SGLang PR Reviewer

把 SGLang 的 merged PR 从“更新日志”变成可验证、可迁移的 AI Infra 工程案例。

**当前版本：v0.2 · Maintainer Lens**

[核心规则](SKILL.md) · [PR 选题](references/pr-selection.md) · [Invariant / Root Cause](references/invariant-and-root-cause.md) · [Impact Surface](references/impact-surface.md) · [History Chain](references/pr-history-chain.md) · [Evidence Ladder](references/evidence-ladder.md) · [Validation Surface](references/validation-surface.md) · [Review Questions](references/review-questions.md) · [类型 Playbook](references/pr-type-playbooks.md) · [审稿 Rubric](references/review-rubric.md)

## v0.2 增加了什么

v0.1 主要解决“我是否真正理解了这个 PR”：Problem → Invariant → Failure Path → Why This Fix → Evidence → Scope → Transfer。

v0.2 再补四个 Maintainer 问题：

```text
这个 PR 从哪里来？          → PR History Chain
它真正影响谁、影响多大？    → Impact Surface
测试究竟覆盖了哪些面？      → Validation Surface
作为 Reviewer 还该问什么？ → Review Questions
```

性能 PR 另外区分 Mechanism、Microbenchmark、Workload、Serving 四层证据，避免把局部 6× 直接写成端到端 6×。

## 为什么需要单独的 PR Reviewer

通用技术写作规则能改善结构和阅读体验，却不能替代 PR 逆向。一个高质量 PR 精读至少要回答：现象是什么、历史 contract 从哪里来、哪个 invariant 被破坏、修改前的数据/状态流在哪里分叉、为什么作者选择这个修法、测试和 benchmark 能证明到哪里、哪些区域没测、哪些结论不能泛化，以及以后 Code Review 时能复用什么问题和规则。

## 什么时候用

- “查下 SGLang 最近 merged PR，挑值得学的。”
- “PR #xxxx 是解决什么问题的？详细教我。”
- “把这个 PR 写成 GitHub 正式技术文章。”
- “Review 这篇 PR 解读，有没有把 scope 或性能讲过头？”
- “这个 PR 的 introducing PR / follow-up 是什么？”
- “这些测试到底覆盖了哪些平台和 failure path？”

如果只是写一篇没有 PR 锚点的 SGLang 源码长文，优先用 `ai-infra-technical-writer`；如果任务是先研究 PR 再成文，可以先用本 Skill 建研究骨架，再用 Technical Writer 做语言与连续叙事精修。

## 默认研究产物

一篇正式 PR 精读通常包含：元信息卡、30 秒结论、History/Impact（确有价值时）、Expected vs Actual、最小背景、Root Cause + Invariant、工程图、Why this fix、Evidence Ladder、Validation Surface、Scope、Review Questions、Bug Pattern 和 Code Review Rule。不是每个元素都必须独立成标题；长文默认只有 5～7 个大章节。

## Worked Examples

- [PR #39871 worked example](examples/pr-39871.md)：Configured Value vs Derived Runtime Value、Single Source of Truth、Observability correctness。
- [PR #39120 worked example](examples/pr-39120.md)：Logical Payload vs Retained Storage、Ownership / Lifetime、Memory Accounting。

正式发布文章见仓库 [SGLang PR 精读](../../docs/pr-reviews/README.md)。

## 当前验证状态

`evals/cases.json` 保留 5 类固定回归用例。v0.2 修改了研究流程和验收字段，但尚未在隔离条件下完整重跑，因此正式通过数仍为 0；详情见 [RESULTS.md](evals/RESULTS.md)。这避免把参与 Skill 设计的公开文章冒充盲测证据。
