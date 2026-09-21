# 正式 PR 精读文章结构

默认 6 个大章节；短 PR 可合并，复杂 PR 原则上不超过 7 个大章。

## 开头

轻量元信息卡：PR、merge commit、改动规模、难度、类型、核心知识、为什么值得读、适用边界。紧接“30 秒结论”，回答：改了行为还是观察？问题根因是什么？最关键知识是什么？

## 六章

1. **问题**：Expected vs Actual；影响是什么；为什么值得关心。
2. **最小背景**：只讲读懂 diff 所需知识；完整概念链接专题文章。
3. **根因**：画修改前失败路径；提炼 invariant。
4. **修复**：按设计动作拆 diff；重点解释 Why this fix 和替代方案。
5. **验证与边界**：Evidence Ladder；Applies to / Does not imply。
6. **迁移**：直接知识 → 通用 Bug Pattern → Code Review Rule → 后续阅读。

## 图解

至少有一张真正解释 Bug 所在位置的工程图，但不机械追求图数。Correctness 常用 Before/After；Performance 用 Timeline；Memory 用 Lifetime；Distributed 用 Rank/Group；Async 用 State Machine；Kernel 用 Shape/Threshold。

## 阅读体验

- 不按 diff 行号逐段翻译；先讲设计意图。
- 不把已有专题文章重新复制一遍；最小背景够用就停。
- 关键源码链接固定到 merge commit。
- 文章能脱离聊天独立阅读，不出现私人工作环境或“我们之前聊过”。
- 不用章节数量、代码块数量或字数代替质量。
