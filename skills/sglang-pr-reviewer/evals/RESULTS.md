# 回归结果

## 2026-09-21 · v0.2

- v0.2 新增 Maintainer Lens：Impact Surface、PR History Chain、Validation Surface、Review Questions，并扩展性能证据层级。
- `cases.json` 仍保留 5 个固定用例，但为每个用例增加 `maintainer_checks`。
- 本次规则升级后尚未在隔离条件下完整重跑 5 个 case，因此 **正式通过数仍为 0**。
- PR #39871 与 #39120 都参与了规则设计，只能作为 worked example / smoke reference，不能当独立盲测证据。
- 下一轮应固定每个 PR 的 source fixture，并让执行者只看到 Skill + fixture + task；保存实际输出后，再按 `must_detect`、`must_not_claim`、`maintainer_checks` 判读。

当前结论：v0.2 的结构和验收维度已落地，但不能宣称已经通过跨类型行为验证。

## v0.1 历史

v0.1 建立了 5 个固定用例与 Invariant / Evidence / Scope 主流程，同样未完成隔离盲测。
