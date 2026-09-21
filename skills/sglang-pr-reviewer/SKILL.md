---
name: sglang-pr-reviewer
description: 筛选、研究、验证和撰写 SGLang merged PR 精读。适用于“最近哪些 PR 值得学”“教我读这个 PR”“把 PR 写成正式技术文章”“Review PR 解读是否过度推断”等任务；重点提炼真实问题、root cause、invariant、Why this fix、Evidence Ladder、Scope 与可迁移 Code Review Rule。不要用于没有 PR/commit 锚点的通用源码长文，也不要把 PR 标题或作者描述当成无需核验的实现事实。
---

# SGLang PR Reviewer

## 目标与边界

把真实 merged PR 从“更新日志”还原成可验证的工程案例：先确定它实际修改了什么，再找出问题、被破坏的 invariant、失败路径、修复设计、证据边界和可迁移的工程规则。最终输出可以是一篇 PR 精读、学习讲解、Review 意见或候选 PR 排序。

本 Skill 不自带 GitHub、源码库、Profiler 或硬件。涉及当前 SGLang 行为时必须读取目标 PR、merge commit 和相关源码；无法访问时缩小结论，不靠模型记忆补齐。PR body、评论、benchmark 是证据，不是绝对真相；实现行为以目标 merge commit 的可达源码和测试为准。

本 Skill 与 `ai-infra-technical-writer` 职责互补：前者负责 PR 逆向与证据，后者擅长长篇叙事与阅读体验。两者都可独立使用；若同时可用，先用本 Skill 建立 PR 研究骨架，再用 Technical Writer 做最终语言与结构精修。

## 从 PR 到结论的工作流

1. **Select**：判断 PR 是否值得精读。优先真实问题清晰、能提炼 invariant、有可迁移知识、diff 可读且有测试/benchmark 的 PR。批量筛选时读取 [PR 选题规则](references/pr-selection.md)。
2. **Freeze Evidence**：固定 `owner/repo`、PR 编号、merge commit、merged time、changed files、diff、相关 tests/benchmark。不要用后来 main 的实现替代 PR 合入时语义；确需补充当前 main 时单独标明。
3. **Reconstruct Problem**：用 Expected vs Actual 还原现象。先回答“正常应该怎样、实际怎样、影响什么”，不要一上来逐行翻译 diff。
4. **Find Invariant**：区分 symptom、proximal cause、root cause 和 invariant。找不到 invariant 时读取 [Invariant 与 Root Cause](references/invariant-and-root-cause.md)。
5. **Trace Failure Path**：按 PR 类型追一条最短失败路径：数据流、状态流、ownership、rank/group、timeline、buffer lifetime、capture/replay contract 等。图只画 Bug 真正发生的位置。
6. **Understand the Fix**：把 diff 按设计动作归纳成 2～5 步，回答“Why this fix”“为什么不选更直接的替代方案”“是否回到已有 Single Source of Truth”。不要把逐行翻译当成理解。
7. **Validate**：区分 Source-confirmed、PR-confirmed、Measured、Inference/Transfer。具体规则见 [Evidence Ladder](references/evidence-ladder.md)。任何 benchmark 都保留模型、硬件、并行配置、shape/并发和测量口径。
8. **Bound the Claim**：明确 Applies to / Does not imply。尤其不要把 CUDA 行为写成 Ascend 行为、把一个模型的阈值泛化成所有模型、把 log-only fix 写成 runtime fix。
9. **Transfer**：提炼一条 Bug Pattern 和一条可复用 Code Review Rule。迁移规则必须比 PR 具体实现更抽象，但不能脱离证据随意泛化。
10. **Write / Review**：正式文章默认使用 6 个大章节，按需加载 [文章结构](references/article-structure.md) 和对应 [PR 类型 Playbook](references/pr-type-playbooks.md)。评分或 95+ 精修时读取 [审稿 Rubric](references/review-rubric.md)。

## 默认输出骨架

正式 PR 精读优先保持 5～7 个大章节，默认六章：

1. 问题：Expected vs Actual，为什么值得关心；
2. 最小背景：只补读懂 diff 所需知识，完整体系链接已有专题；
3. 根因：修改前失败路径与被破坏的 invariant；
4. 修复：按设计意图拆 diff，并解释 Why this fix；
5. 验证与边界：Evidence Ladder、Scope、Does not imply；
6. 迁移：直接知识、通用 Bug Pattern、Code Review Rule、后续阅读。

文章开头默认给一个轻量元信息卡：PR、merge commit、改动规模、难度、类型、核心知识、为什么值得读、适用边界；再给“30 秒结论”。短答或纯教学问答不强制套用完整结构。

## PR 类型与按需检查

固定骨架统一，但分析重点随类型变化。Correctness 看 invariant 和状态语义；Memory 看 tensor lifetime / peak accounting；Performance 看 timeline / critical path / benchmark methodology；Distributed 看 rank / group / ownership / collective；Scheduler 看 state transition / queue / terminal state；Speculative 看 Draft / Verify / Accept / Commit contract；KV Cache 看 allocate / write / publish / reclaim；Kernel 看 shape / dtype / tiling / launch cost / fallback；Graph 看 capture / replay contract；CI/Infra 看 predicate / coverage / false positive / false negative。完整表见 [PR 类型 Playbook](references/pr-type-playbooks.md)。

## 证据与禁止事项

- 不根据 PR 标题推断实现；必须读取 diff 和关键上下文。
- 不把 PR 作者的 motivation 当成源码事实；两者冲突时对实现行为以目标代码为准，并说明差异。
- 不编造 benchmark、Profiler、CI、成功日志、硬件结果或设计动机。
- 不把最新 main 的新抽象反写成旧 merge commit 当时已经存在的实现。
- 不因一个平台 PR 就跨平台泛化；GPU/CUDA、ROCm、Ascend/NPU、XPU 的路径分别核验。
- 不因文章需要“完整”而扩展到与 PR 无关的整套框架教程；已有专题用链接承接。
- 不为了高分机械增加章节、源码层数和图；新增内容必须降低理解成本或提高证据强度。

## Review 与评分

评分按 100 分制只作编辑与研究判断，不是客观测量。技术准确性、证据与版本、Root Cause/Invariant 是硬门槛。存在关键源码错误、版本/硬件路径混用时不得用文风高分抵消；把 inference 写成 confirmed fact 时不得进入标杆稿。完整权重和 blocking rules 见 [审稿 Rubric](references/review-rubric.md)。

## 回归

本 Skill 初始回归集见 [evals/cases.json](evals/cases.json)，覆盖 Correctness、Async State Machine、Memory、Kernel Performance、PD Communication 五类 PR。修改 Skill 后先保存实际输出，再按每个 case 的 `must_detect` / `must_not_claim` 判定。当前验证状态见 [evals/RESULTS.md](evals/RESULTS.md)。
