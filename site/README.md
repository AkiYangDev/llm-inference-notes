# 网站维护

本站由 `docs/` 中已发布的 Markdown 生成，文章只维护一份源文件。推送到 main 后，GitHub Actions 构建并部署至 GitHub Pages；项目路径为 `/llm-inference-notes/`。

## 构建与预览

需要 Python 3.12+、Node.js 22.13+ 和 pnpm 11.25.0。Linux 构建机还需要中文字体，Ubuntu 可安装 `fonts-noto-cjk`；Windows 使用系统微软雅黑，macOS 使用苹方。字体只在构建分享卡片时使用，读者无需下载中文字体。

在仓库根目录执行：

```bash
python3 -m pip install -r site/requirements.txt
npm install -g pnpm@11.25.0
pnpm install --frozen-lockfile --ignore-scripts
python3 site/build.py
python3 scripts/serve_site.py --port 8876
```

打开 `http://127.0.0.1:8876/llm-inference-notes/`。预览只监听本机回环地址；修改后重新构建并刷新。生成目录 `_site/` 和 `node_modules/` 不提交。

构建末尾运行 `scripts/enhance_site.mjs`：代码高亮和 KaTeX 公式生成静态 HTML，Mermaid 及其分块、公式样式和字体随站点托管。依赖版本固定在 `package.json` 和 `pnpm-lock.yaml`，构建时从包管理仓库安装，读者打开文章不依赖第三方 CDN。开源许可随生成资源保留。

## 内容入口与元数据

- `/articles/` 为完整目录，支持专题、层级、标签、目录内筛选、排序和稍后读。
- `/guides/` 根据读者当前问题组织推荐文章；`site/discovery.json` 维护四条路线和首页精选，指向未发布文件会让构建失败。
- `/series/` 展示完整系列，`site/series.json` 维护顺序、简短标题和介绍。
- `site/articles.json` 维护每篇文章的摘要、专题、标签和阅读层级；文章缺少元数据时构建失败。

四种层级为入门、进阶、深入研究和速查，对应 `entry`、`advanced`、`deep`、`reference`。层级帮助选择阅读深度，专题用于归档，系列说明推荐顺序。阅读地图不是运行时调用图。

首页将精选入口与最新发表的三篇文章分开展示；既有文章重写不会被当作新文章。文章列表保留最近更新排序。

`seo_title` 和 `seo_summary` 可选，只改变搜索结果标题与摘要，不改变正文 H1。阅读时长按中文、英文、代码行和图表粗估；不包括实际运行命令的时间，也不是完成阅读的承诺。

## 长文阅读与搜索

桌面使用右侧目录；手机保留文首折叠目录，并提供随时打开的“本页目录”。选择章节后关闭弹窗，定位对应标题。搜索或外部链接指向折叠的实现细节时，会先展开祖先区块再定位。

宽表格保留局部横向滚动，实际溢出时显示提示；正文配图由本站托管，按需加载。Mermaid 和链接到图片文件的正文配图可放大、缩小及适应窗口，Esc 关闭后恢复焦点。禁用 JavaScript 时正文、高亮代码、公式及图片原链接仍可阅读；Mermaid 保留源码描述。

搜索按文章与章节建立索引，每篇只显示一个最佳结果。标题优先，再看章节标题和正文；结果可定位章节。中英文术语别名统一常见写法，如“首字延迟 / TTFT”“KVCache / KV Cache”“PD分离 / PD 分离”；TPOT 与 ITL、AL 与接受率保持独立。首次显示 15 篇，可继续加载其余结果。目录里的关键词可直接带入全文搜索。

`/` 或 `Ctrl/⌘ + K` 打开搜索；`Alt + Shift + ←/→` 切换系列前后篇。输入、选择文本或弹窗打开时不触发系列跳转。

## 阅读记录与主题

主题默认跟随系统，手动切换后保留偏好，Mermaid 随主题重绘。字号支持 16–22px，并提供专注阅读、代码复制、代码行链接和章节链接复制。

已读标记、稍后读、字号与阅读位置只存于当前浏览器。不会根据滚动到底自动标记已读；用户在文末主动标记，可撤销。首页在存在记录时显示继续阅读入口。不登录、不跨设备同步，清除站点数据会删除记录。网站没有访问统计或跟踪脚本。

## 发布、分享与搜索收录

构建自动生成 Sitemap、RSS、canonical、Open Graph 与 JSON-LD。每篇文章生成一张包含真实标题、专题和摘要的 1200 × 630 PNG 分享卡，标题或摘要变化后图片 URL 随内容哈希更新。CSS、入口脚本和搜索模块也使用内容版本文件名，避免发布后混用旧模块。

文章首次提交与最近更新读取完整 Git 历史，首次提交不等同于首次公开发布日期。源码依据只提取导读中明确记录的 `owner/repo @ 40位 commit`，不会自动跟随上游。简短的更新日期显示在文章头部，完整依据和修改记录可展开。

旧地址由 `site/redirects.json` 维护。GitHub Pages 不提供本项目自定义 HTTP 301，因此生成带 canonical 的静态跳转页，并保留查询参数和章节锚点。旧代码行链接在文章结构变化后可能需要重新定位。

首页、正文、专题、阅读路线和层级归档允许索引；404、旧地址跳转和细粒度标签页为 `noindex,follow`。标签页不进入 Sitemap。

Google Search Console 操作：

1. 添加网址前缀属性 `https://akiyangdev.github.io/llm-inference-notes/`。
2. 将 HTML 标记验证的 content 值填入 `site/search-console.json` 中的 `google_site_verification`，部署后验证。空值不输出标签。
3. 提交 `https://akiyangdev.github.io/llm-inference-notes/sitemap.xml`。
4. 用网址检查核对首页和代表文章的实际收录状态。提交不保证收录或排名。

有效的 robots.txt 位于域名根目录 `/robots.txt`，本项目子目录中的同名文件不能控制全站。参考 [Google Sitemap 文档](https://developers.google.com/search/docs/crawling-indexing/sitemaps/build-sitemap) 和[所有权验证说明](https://support.google.com/webmasters/answer/9008080)。

## Skills 与素材

`skills/<slug>/README.md` 与同目录 `SKILL.md` 同时存在时，自动生成 `/skills/<slug>/`；首页、目录及搜索收录介绍页。完整规则、参考资料和评测仍链接到 GitHub，不计入技术文章数量。

首页使用所有者提供的完整鲸鱼娘立绘和 Q 版动画。手机端缩小立绘并与文字并排展示。视频只在可视区域加载与静音循环，离开区域或切至后台暂停；减少动态效果时先显示封面，可手动播放。角色与衍生素材来源见 [SOURCES.md](assets/SOURCES.md) 和网站 `/credits/`，历史素材继续存档。本站为独立个人技术站，非 DeepSeek 官方。

## 发布前检查

```bash
python3 scripts/check_docs.py --strict-style
python3 scripts/check_math.py
python3 site/build.py
python3 scripts/check_site.py
node scripts/check_search.mjs
node --input-type=module --check < site/assets/app.js
```

PowerShell 中最后一条改用 `Get-Content -Raw site/assets/app.js | node --input-type=module --check`。

检查覆盖本地链接、站内锚点、分类归属、SEO 元数据、分享卡尺寸、静态公式与高亮资源、搜索别名和旧地址跳转。浏览器还需检查手机与桌面布局、暗色模式、搜索与目录弹窗、图表放大、键盘焦点、减少动态效果和禁用 JavaScript 的阅读体验；自动检查不证明技术文章结论或搜索引擎收录状态。
