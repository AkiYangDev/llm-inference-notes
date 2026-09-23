# 网站维护

网站由 `docs/` 中除目录 README 以外的 Markdown 文章生成，文章不需要维护第二份 HTML。添加文章后，推送到 main 会运行构建并部署至 GitHub Pages。

## 本地构建

在仓库根目录执行：

```bash
python3 -m pip install -r site/requirements.txt
python3 site/build.py
```

生成目录为 `_site/`。站点基础路径为 `/llm-inference-notes/`，部署路径变化时需要同步调整 `site/build.py` 的 BASE。

首页高质量看板图由 `site/asset-payloads/whale-hero-v2/` 的文本分片在构建时还原为单张 `whale-hero-v2.webp`；`site/build.py` 会校验长度和 SHA-256，分片缺失或损坏会让构建直接失败。

## 界面资源

配色与响应式布局位于 `site/assets/style.css`，搜索、主题切换、复制与图表放大位于 `site/assets/app.js`。搜索索引从文章正文生成。

Mermaid、KaTeX 公式与代码高亮按需使用固定版本的 jsDelivr 模块；网络不可用时保留可阅读正文和源码。字体加载失败时使用系统字体。网站没有访问统计或跟踪脚本；本地存储用于主题偏好与当前浏览器的阅读进度。

角色图片为用户指定的 DeepSeek 鲸鱼娘主题插画，原始链接见 [素材来源](assets/SOURCES.md)。本站为独立个人技术站，不代表 DeepSeek 官方。

## 阅读与导航

- `/articles/` 汇总全部已发布文章；`/topics/<topic>/` 按实际有文章的专题生成目录，不展示空占位栏目。
- 搜索按文章与章节生成索引，结果高亮关键词并可跳到对应标题。正文标题 ID 由 Markdown TOC 扩展生成。
- 主题默认跟随系统，手动切换后保留偏好；Mermaid 图随深浅色模式重绘。
- 桌面提供右侧目录，窄屏提供折叠目录；表格可横向滚动，工程图支持缩放。
- `/credits/` 记录插画来源。首页展示 WebP 透明背景素材，原始提供图仍保留。
- 样式与交互脚本按内容哈希更新版本，搜索索引使用重新验证的请求。

## 内容四层

每篇公开文章都在 `site/articles.json` 中声明一个 `role`，用于区分文章在增长与阅读路径中的主要职责，而不是替代专题分类。

- `entry` / 入门：回答更广泛的搜索问题，降低第一次进入 AI Infra / SGLang 的阅读门槛。
- `advanced` / 进阶：从概念进入实现，连接 Scheduler、ModelRunner、Attention、MoE、分布式与量化执行。
- `deep` / 深入研究：源码、性能、投机解码、Ascend Profiling 与 PR 级工程细节。
- `reference` / 速查：适合反复查阅的名词表、Rank / Group、Collective 与知识地图。

`/articles/` 默认按这四层组织，并生成 `/levels/entry/`、`/levels/advanced/`、`/levels/deep/`、`/levels/reference/` 四个独立归档页。首页同时提供层级入口；专题、标签和系列阅读仍保留，四层结构与原有信息架构并行存在。

文章可选声明 `seo_title` 与 `seo_summary`。它们只服务浏览器标题与搜索摘要，不改正文 H1，因此可以在不牺牲技术标题准确性的情况下，为高价值入口页提供更短、更明确的搜索结果文案。所有公开 Markdown 必须在 `site/articles.json` 有编辑元数据，否则构建失败，避免新文章悄悄退化到默认 SEO 信息。

## Skills 栏目

网站自动发现 `skills/<slug>/README.md`，同目录必须存在 `SKILL.md`。

- `/skills/` 生成目录，`/skills/<slug>/` 展示对应 README 全文。
- README 的一级标题作为名称，首段作为简介；根 `skills/README.md` 的首段作为栏目介绍。
- 首页和导航提供入口，搜索同时覆盖文章与 Skill 介绍，并标注内容类型。
- 已有网站页面之间的相对链接转换为站内链接；规则、参考资料、评测与示例仍链接到 GitHub 对应文件。
- 修改介绍或新增满足以上结构的 Skill 后，提交到 main 即自动更新，无需手工维护网页或列表。

完整规则不作为单独的博客文章导入，也不计入技术文章数量。

## 系列阅读

`site/series.json` 定义系列名称、专题、文章路径和阅读顺序。构建只收录 `docs/` 中实际存在且有一级标题的文章，过滤缺失项后再生成编号、当前篇目、上一篇与下一篇。未发布的配置项不会出现在页面，也不会生成占位或失效链接。

SGLang 系列顺序为请求执行全链路、Scheduler、ModelRunner、Attention 与 KV Cache、KV Cache 专题。按配置路径发布新文章后，会自动加入系列；如果文章使用其他文件名，需同步修改配置中的路径。

首页阅读地图、独立 `/series/` 阅读中心、文章目录和 SGLang 专题页展示已发布系列；文章顶部提供系列导航，底部提供相邻文章。首篇不显示上一篇，末篇不显示下一篇；Skills 不参与文章系列编号。

阅读地图用编号展示推荐阅读顺序，不表示运行时调用关系。文末的「标记为已读」可撤销；地图显示已读数量，并链接到首篇未读文章。进度保存在当前浏览器，不跨设备同步；未读完不会自动标记完成。

## 发布与阅读功能

构建自动生成 `sitemap.xml`、`feed.xml`、Open Graph 与 JSON-LD。分享卡片使用站点封面，标题和摘要按页面生成。

文章首次提交与最近更新读取完整 Git 历史；首次提交不等同于首次公开发布日期。源码依据仅提取文章导读中明确记录的 `owner/repo @ 40位 commit`，更新文章标注后随构建同步，不自动跟随上游版本。

点击代码行号可高亮并分享该行链接；文章结构改动可能使旧的代码块编号发生变化。正文源码引用保留固定版本链接。阅读位置保存在当前浏览器，重新打开时点击“继续上次阅读”恢复。Alt + 左右方向键切换系列前后篇，输入框、选中文本和弹窗打开时不触发。

## Google Search Console

网站已输出完整静态正文、canonical、Sitemap、RSS 与 JSON-LD。首页、正文、专题页与四层阅读归档允许索引；404 与标签筛选页保持 `noindex,follow`。标签页主要承担站内导航，当前标签粒度较细，不进入 Sitemap，避免大量单篇/双篇薄归档稀释年轻站点的索引信号。发布前运行 `python scripts/check_site.py` 检查元数据、站内链接和 Sitemap。

1. 在 Google Search Console 添加网址前缀属性 `https://akiyangdev.github.io/llm-inference-notes/`。
2. 选择 HTML 标记验证，将 Google 提供的 content 值填入 `site/search-console.json` 的 `google_site_verification`，提交并等待部署后验证。空配置不会输出伪造标签。
3. 提交 `https://akiyangdev.github.io/llm-inference-notes/sitemap.xml`。
4. 使用网址检查查看首页和代表文章，按需请求编入索引。提交不保证收录或排名。

这是 GitHub Pages 的项目子目录。有效的 robots.txt 必须位于域名根目录 `/robots.txt`；不要把本项目内的 robots.txt 当作有效全站规则。无 robots.txt 不等于禁止抓取，本项目通过 Search Console 提交 Sitemap。

参考：[Google Sitemap 文档](https://developers.google.com/search/docs/crawling-indexing/sitemaps/build-sitemap)、[所有权验证](https://support.google.com/webmasters/answer/9008080)。

## 动效

首屏分层淡入，卡片与按钮轻量反馈，阅读地图按已读数量更新进度条，搜索框短促淡入。桌面光晕仅循环两轮，离开首屏或隐藏标签页时暂停；移动端取消光晕呼吸。减少动态效果设置关闭全部动效，正文不使用滚动显现。


## 编辑部视觉系统

首页采用雾白、紫蓝与蓝白鲸鱼娘立绘，最新文章保持自动生成；完整阅读地图位于 `/series/`，首页用简洁入口连接。手机端首屏上下排列，正文延续独立阅读排版。

`whale-hero-v2.webp` 为首页宽幅看板图：桌面端整幅铺入 Hero，并在左侧叠加阅读友好的渐变；移动端使用同一资源做右侧焦点裁切，避免再下载一张移动端大图。`whale-states.webp` 仍为 Q 版四宫格，CSS 的 `.mascot-*` 用于阅读入口、搜索、完成操作与 404；`whale-editor.webp` 保留为此前首屏版本的历史素材。素材来源页区分角色设计与 AI 延展素材。减少动态效果设置关闭动画与过渡。

## 阅读体验与回归检查

文章目录直接展示文章；全量标签默认折叠，阅读地图使用单独入口。手机导航保留文章、阅读地图、Skills、搜索与主题切换。对外层级名称为「入门 / 进阶 / 深入研究 / 速查」，内部 role 标识和已发布 URL 保持稳定。

搜索支持空格分隔的多个关键词，每篇内容只显示一个最佳结果；标题优先，其次章节标题，再次正文。章节命中继续跳到具体章节，结果数按文章和 Skill 数量统计。

独占行的 `$$` 和 `\[` / `\]` 公式块在 Markdown 解析前保留原始 TeX；代码围栏里的示例不参与公式渲染。KaTeX 加载失败时保留公式源码。旧文章地址的脚本跳转保留查询参数与章节锚点，无 JavaScript 时仍提供跳转和手动链接。

PR 检查与正式发布均运行文档、站点构建、站内锚点与静态资源、数学公式、搜索排序及 JavaScript 语法检查：

```bash
python3 scripts/check_docs.py
python3 scripts/check_math.py
python3 site/build.py
python3 scripts/check_site.py
node scripts/check_search.mjs
node --input-type=module --check < site/assets/app.js
```

这些检查不能代替真实浏览器的视觉、交互、CDN 可达性与 Search Console 收录验证。
