# SGLang PR Reviewer

把 SGLang 的 merged PR 从“更新日志”变成可验证、可迁移的 AI Infra 工程案例。

[核心规则](SKILL.md) · [PR 选题](references/pr-selection.md) · [Evidence Ladder](references/evidence-ladder.md) · [类型 Playbook](references/pr-type-playbooks.md) · [审稿 Rubric](references/review-rubric.md) · [回归用例](evals/cases.json)

## 为什么需要单独的 PR Reviewer

通用技术写作规则能改善结构和阅读体验，却不能替代 PR 逆向。一个高质量 PR 精读至少要回答：现象是什么、哪个 invariant 被破坏、修改前的数据/状态流在哪里分叉、为什么作者选择这个修法、测试和 benchmark 能证明到哪里、哪些结论不能泛化，以及以后 Code Review 时能复用什么检查规则。

这个 Skill 专门处理这些问题。它不鼓励“最近合了什么就写什么”，而是先筛选学习价值，再冻结 merge commit 的证据，最后把真实 patch 提炼成工程知识。

## 什么时候用

- “查下 SGLang 最近 merged PR，挑值得学的。”
- “PR #xxxx 是解决什么问题的？详细教我。”
- “把这个 PR 写成 GitHub 正式技术文章。”
- “Review 这篇 PR 解读，有没有把 scope 讲过头？”
- “这几个 PR 哪些适合初学者，哪些适合学性能/分布式？”

如果只是写一篇没有 PR 锚点的 SGLang 源码长文，优先用 `ai-infra-technical-writer`；如果任务是先研究 PR 再成文，可以先用本 Skill 建研究骨架，再用 Technical Writer 做语言与连续叙事精修。

## 默认研究产物

一篇正式 PR 精读通常包含：元信息卡、30 秒结论、Expected vs Actual、最小背景、Root Cause + Invariant、Before/After 工程图、Why this fix、Evidence Ladder、Scope、Bug Pattern 和 Code Review Rule。不是每个元素都必须独立成标题；长文默认只有 5～7 个大章节。

## Worked Example

[PR #39871 worked example](examples/pr-39871.md) 展示如何把一个只有 1 file / +8 / -2 的日志修复，提炼成 `Configured Value vs Derived Runtime Value`、Single Source of Truth 和 Observability correctness。正式发布文章见 [《PR #39871 为什么 TP16 不是 AttnTP16？》](../../docs/pr-reviews/sglang-pr-39871-attention-parallel-widths.md)。

## 当前验证状态

初始规则已用 PR #39871 的公开文章做设计校对，但这不是独立盲测，不计入回归通过数。`evals/cases.json` 已准备 5 类固定用例，尚未在隔离条件下完整重跑；详情见 [RESULTS.md](evals/RESULTS.md)。因此当前定位是可使用、待持续回归的 v0.1 方法 Skill，不以自评分数宣称普遍效果。
