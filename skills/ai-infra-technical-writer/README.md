# AI Infra Technical Writer

面向有后端开发经验、正在阅读和实践 AI Infra 的工程师，将源码分析写成连贯、可查证的技术文章。

适用于 SGLang、vLLM、DeepSeek、Ascend NPU、KV Cache、模型执行、分布式推理与性能分析。默认用中文写作，保留英文技术标识符；模型案例优先 DeepSeek，并核对具体代际与实现。

## 它解决什么问题

技术文章容易出现三种断点：类名列得很多，却没有解释数据如何流动；数学 shape 正确，却接不上真实算子布局；引用看似齐全，却混用了模型、版本或后端。

这个 Skill 要求先取证，再围绕同一请求或 Tensor 展开叙事。长篇默认采用 5–7 个大章节，图表各自回答一个具体问题；准确性是硬约束，不能为了顺畅而省去决定结论的条件。

## 核心规则

- **证据先行**：确认版本、入口、调用方和分支条件，引用固定版本源码。
- **主线连续**：追踪请求、batch、Tensor 和缓存状态的变化，先解释结果来源，再使用它。
- **模型明确**：优先 DeepSeek；更换模型时重新核对层内结构、缓存、shape 和算子，不只替换名字。
- **布局接通**：区分数学维度、打包布局、局部 head、padding 与算子实参。
- **图文互补**：图解释关系与顺序，表解释精确映射，正文解释原因与变化。
- **边界清楚**：公开源码分析、部署环境确认和实机测量各有证据范围；不编造成功结果或性能数字。

## 使用方式

将完整的 `ai-infra-technical-writer` 目录交给支持 Skill 的工具导入，保留 `SKILL.md` 与 `references/` 的相对路径。具体安装位置以所用工具为准。

不支持 Skill 的工具，也可以让 Agent 读取 `SKILL.md`，并按任务读取它引用的参考文件。只有规则文件不够：源码类任务仍需要可访问的目标仓库，部署或性能文章仍需要相应日志与测量材料。

可以直接使用以下请求：

```text
使用 ai-infra-technical-writer 撰写一篇 SGLang 请求执行链路文章。
以 DeepSeek 为案例，先核对具体模型版本与 Ascend 后端源码。
面向有后端经验的读者，用同一请求贯穿调度、模型、缓存和返回路径，
提供可定位的来源与必要工程图，并最终输出 Markdown。
```

审核已有文章时：

```text
使用 ai-infra-technical-writer 审核这篇文章。
先核对技术结论和适用范围，再检查连续阅读体验。
指出有具体段落依据的问题；将源码核对、实机验证和编辑评分分开。
```

## 内容结构

| 文件 | 作用 |
|---|---|
| [SKILL.md](SKILL.md) | 触发范围、工作流程与输出要求 |
| [source-and-deployment.md](references/source-and-deployment.md) | 源码取证与部署文章边界 |
| [execution-chain-review.md](references/execution-chain-review.md) | 调用链、设备边界、模型迁移与布局检查 |
| [reading-experience.md](references/reading-experience.md) | 连续阅读、术语解释、图表与代码安排 |
| [article-assessment.md](references/article-assessment.md) | 有依据的文章评分与审稿 |
| [engineering-evidence-to-outcomes.md](references/engineering-evidence-to-outcomes.md) | 从真实工程证据形成分享与项目材料 |

## 示例与维护

[DeepSeek-V4 / SGLang 执行链路文章](../../docs/sglang/sglang-ascend-request-lifecycle.md) 展示了这些规则如何用于请求、模型、缓存与设备算子的解释。

迭代时，从具体失败段落提炼规则，再用新的任务或独立审阅检验。新增规则应能迁移到其他文章；一次自评提高不等于质量已被客观证明，也不保证后续文章全部正确。

本目录发布的是可独立使用的核心规则与本项目维护的参考资料。个人安装包中的第三方扩展模块、示例集合和平台专用资源没有随此目录分发。源码优先取证、具体示例和图解等写作思路参考了 [vllm-technical-blog-writer](https://github.com/shen-shanshan/vllm-dev-skills/blob/master/skills/vllm-technical-blog-writer/GUIDE.md) 等资料。第三方模块保留各自的来源和许可，不包含在此公开目录中。

更新日期：2026-09-20。
