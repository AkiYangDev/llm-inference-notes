---
name: sglang-pr-reviewer
description: 筛选、研究、验证和撰写 SGLang merged PR 精读。适用于“最近哪些 PR 值得学”“教我读这个 PR”“把 PR 写成正式技术文章”“Review PR 解读是否过度推断”等任务；重点提炼真实问题、Impact Surface、PR History、root cause、invariant、Why this fix、Evidence / Validation Surface、Scope、Review Questions 与可迁移 Code Review Rule。不要用于没有 PR/commit 锚点的通用源码长文，也不要把 PR 标题、作者描述或单次 benchmark 当成无需核验的实现事实。
---

# SGLang PR Reviewer

> v0.2：在 v0.1 的 Invariant / Evidence / Scope 基础上，增加 Maintainer 视角的 Impact Surface、PR History Chain、Validation Surface 与 Review Questions。

## 目标与边界

把真实 merged PR 从“更新日志”还原成可验证的工程案例：不仅解释改了什么，还回答它从哪里来、在什么条件下触发、影响谁、破坏了什么 invariant、为什么这样修、测试覆盖了哪些面、哪些面仍未验证，以及作为 Reviewer 下一步还应该追问什么。

本 Skill 不自带 GitHub、源码库、Profiler 或硬件。涉及当前 SGLang 行为时必须读取目标 PR、merge commit 和相关源码；无法访问时缩小结论，不靠模型记忆补齐。PR body、评论、benchmark、CI 都是证据，但证据类型和覆盖面不同；实现行为以目标 merge commit 的可达源码和测试为准。

本 Skill 与 `ai-infra-technical-writer` 职责互补：本 Skill 负责 PR 逆向、证据和 Reviewer 判断，Technical Writer 擅长长篇叙事与阅读体验。两者都可独立使用；若同时可用，先用本 Skill 建立研究骨架，再用 Technical Writer 做最终语言与结构精修。

## 从 PR 到结论的工作流

1. **Select**：判断 PR 是否值得精读。优先问题清晰、能提炼 invariant、有可迁移知识、diff 可读且有测试/benchmark 的 PR。批量筛选时读取 [PR 选题规则](references/pr-selection.md)。
2. **Freeze Evidence**：固定 `owner/repo`、PR 编号、merge commit、merged time、changed files、diff、tests/benchmark。不要用后来 main 的实现替代 PR 合入时语义；确需补充当前 main 时单独标明。
3. **Reconstruct History + Impact**：先看 Issue、introducing PR、stacked-on / depends-on、follow-up，再写 Trigger、Symptom、Affected Config、User-visible Impact、Blast Radius。需要时读取 [PR History Chain](references/pr-history-chain.md) 和 [Impact Surface](references/impact-surface.md)。
4. **Reconstruct Problem**：用 Expected vs Actual 还原现象。先回答“正常应该怎样、实际怎样、为什么值得修”，不要一上来逐行翻译 diff。
5. **Find Invariant**：区分 symptom、proximal cause、root cause 和 invariant。找不到 invariant 时读取 [Invariant 与 Root Cause](references/invariant-and-root-cause.md)。
6. **Trace Failure Path**：按 PR 类型追一条最短失败路径：数据流、状态流、ownership、rank/group、timeline、buffer lifetime、capture/replay contract 等。图只画 Bug 真正发生的位置。
7. **Understand the Fix**：把 diff 按设计动作归纳成 2～5 步，回答 Why this fix、为什么不选更直接方案、是否回到已有 Single Source of Truth。不要把逐行翻译当成理解。
8. **Validate**：一边做 [Evidence Ladder](references/evidence-ladder.md)，一边做 [Validation Surface](references/validation-surface.md)。不仅记录“测过什么”，还记录平台、shape、并发、失败路径与未覆盖区域。
9. **Bound the Claim**：明确 Applies to / Does not imply。尤其不要把 CUDA 行为写成 Ascend 行为、把一个模型/shape 的阈值泛化成所有 workload、把机制收益直接升级成端到端收益。
10. **Review + Transfer**：先生成 [Review Questions](references/review-questions.md)，再提炼 Bug Pattern 与可复用 Code Review Rule。规则必须比当前 PR 更抽象，但不能脱离证据随意泛化。
11. **Write / Review**：正式文章默认使用 6 个大章节，按需加载 [文章结构](references/article-structure.md) 和对应 [PR 类型 Playbook](references/pr-type-playbooks.md)。评分或 95+ 精修时读取 [审稿 Rubric](references/review-rubric.md)。

## 默认输出骨架

正式 PR 精读优先保持 5～7 个大章节，默认六章：

1. 问题：Expected vs Actual，并把 Trigger / Impact 放到读者能快速看见的位置；
2. 最小背景：只补读懂 diff 所需知识，完整体系链接已有专题；
3. 根因：修改前失败路径与被破坏的 invariant，必要时补 History Chain；
4. 修复：按设计意图拆 diff，并解释 Why this fix；
5. 验证与边界：Evidence Ladder、Validation Surface、Scope、Does not imply；
6. 迁移：直接知识、通用 Bug Pattern、Review Questions、Code Review Rule、后续阅读。

文章开头默认给轻量元信息卡：PR、merge commit、改动规模、难度、类型、核心知识、为什么值得读、适用边界；再给“30 秒结论”。短答、候选筛选和纯教学问答不强制套完整文章结构。

## PR 类型与按需检查

固定骨架统一，但分析重点随类型变化。Correctness 看 invariant / semantic drift；Memory 看 ownership / lifetime / peak accounting；Performance 看 timeline / critical path / performance evidence level；Distributed 看 rank / group / ownership / collective；Scheduler 看 state transition / queue / terminal state；Speculative 看 Draft / Verify / Accept / Commit contract；KV Cache 看 allocate / write / publish / reclaim；Kernel 看 shape / dtype / tiling / launch / fallback；Graph 看 capture / replay contract；CI/Infra 看 predicate / coverage / false positive / false negative。完整表见 [PR 类型 Playbook](references/pr-type-playbooks.md)。

## 证据与禁止事项

- 不根据 PR 标题推断实现；必须读取 diff 和关键上下文。
- 不把 PR 作者的 motivation 当成源码事实；两者冲突时对实现行为以目标代码为准，并说明差异。
- 不编造 benchmark、Profiler、CI、成功日志、硬件结果或设计动机。
- 不把最新 main 的新抽象反写成旧 merge commit 当时已经存在的实现。
- 不因 follow-up PR 的后来修复，倒推出当前 PR 当时已经满足同一 contract。
- 不因一个平台 PR 就跨平台泛化；GPU/CUDA、ROCm、Ascend/NPU、XPU 的路径分别核验。
- 不把 mechanism improvement、microbenchmark、model workload、serving throughput 当成同一层性能证据。
- 不因文章需要“完整”而扩展到与 PR 无关的整套框架教程；已有专题用链接承接。
- 不为了高分机械增加章节、源码层数和图；新增内容必须降低理解成本或提高证据强度。

## Review 与评分

评分按 100 分制只作编辑与研究判断，不是客观测量。技术准确性、证据与版本、Root Cause/Invariant 是硬门槛。v0.2 额外检查 Impact 是否具体、History 是否影响因果解释、Validation Surface 是否暴露未测区域、Review Questions 是否能指导下一次审查。完整权重和 blocking rules 见 [审稿 Rubric](references/review-rubric.md)。

## 回归

回归集见 [evals/cases.json](evals/cases.json)。修改 Skill 后先保存实际输出，再按 `must_detect` / `must_not_claim` 和 v0.2 的 maintainer checks 判定。当前验证状态见 [evals/RESULTS.md](evals/RESULTS.md)。
