# PR 精读审稿 Rubric

总分 100，只用于编辑与研究判断。v0.2 将“Maintainer 视角”并入原维度，而不是靠新增大量同义分项凑分。

| 维度 | 分值 | 核心验收 |
| --- | ---: | --- |
| 技术准确性 | 20 | 核心实现、条件、公式无关键错误 |
| 证据与版本 | 15 | 固定 merge commit；History 不串版本；来源支持对应断言 |
| Root Cause / Invariant | 15 | 不停在 symptom；History/Failure Path 能支撑稳定 invariant |
| Fix / Why This Fix | 10 | 按设计意图解释，并比较合理替代方案 |
| Scope / Impact / Boundary | 10 | Trigger、Affected Config、Blast Radius 与 Does not imply 清楚 |
| 教学与认知负担 | 10 | 最小背景、例子和术语引入适合目标读者 |
| 阅读体验 | 8 | 5～7 大章，连续叙事，无明显重复/碎片化 |
| 图解 | 5 | 图回答真实工程问题，图文语义一致 |
| Validation / Transfer | 5 | 暴露未测 surface，并给出可复用 Review Questions / Rule |
| 系列协同 | 2 | 与已有专题互链，不重复造一套知识体系 |

## Blocking rules

- 存在关键源码错误、主调用路径不成立、硬件路径写反：公开发布就绪分不得进入 90+。
- 混用版本、用 current main 冒充 merge commit，或让 follow-up 能力倒灌到旧 PR：不得进入 90+。
- 把 Inference / Transfer 写成 Source-confirmed：不得进入 95+。
- 把 mechanism / microbenchmark 收益直接写成 serving 收益：不得进入 95+。
- 标题承诺的核心问题没有被回答或没有证据：不得进入 90+。
- CI green 但目标测试实际 skip，仍写“验证通过”：视为证据错误。

## 95+ 与标杆稿

95+ 应同时满足：核心事实可追溯、invariant 清楚、Why this fix 有深度、Impact 具体、Validation Surface 暴露空白、边界准确、阅读顺畅，并至少给出一组能用于陌生代码的 Review Questions。所谓“98 分”不靠多加章节实现，而是要求几乎没有实质性技术/证据缺口且表达足够精炼。自评分不是用户测试，也不是客观质量证明。
