# 网站维护

网站由 `docs/` 中除目录 README 以外的 Markdown 文章生成，文章不需要维护第二份 HTML。添加文章后，推送到 main 会运行构建并部署至 GitHub Pages。

## 本地构建

在仓库根目录执行：

```bash
python3 -m pip install -r site/requirements.txt
python3 site/build.py
```

生成目录为 `_site/`。站点基础路径为 `/llm-inference-notes/`，部署路径变化时需要同步调整 `site/build.py` 的 BASE。

## 界面资源

配色与响应式布局位于 `site/assets/style.css`，搜索、主题切换、复制与图表放大位于 `site/assets/app.js`。搜索索引从文章正文生成。

Mermaid 与代码高亮使用固定版本的 jsDelivr 模块；网络不可用时保留可阅读正文和源码。字体加载失败时使用系统字体。网站没有访问统计或跟踪脚本；本地存储仅用于主题偏好。

角色图片为用户指定的 DeepSeek 鲸鱼娘主题插画，原始链接见 [素材来源](assets/SOURCES.md)。本站为独立个人技术站，不代表 DeepSeek 官方。

## 阅读与导航

- `/articles/` 汇总全部已发布文章；`/topics/<topic>/` 按实际有文章的专题生成目录，不展示空占位栏目。
- 搜索按文章与章节生成索引，结果高亮关键词并可跳到对应标题。正文标题 ID 由 Markdown TOC 扩展生成。
- 主题默认跟随系统，手动切换后保留偏好；Mermaid 图随深浅色模式重绘。
- 桌面提供右侧目录，窄屏提供折叠目录；表格可横向滚动，工程图支持缩放。
- `/credits/` 记录插画来源。首页展示 WebP 透明背景素材，原始提供图仍保留。
- 样式与交互脚本按内容哈希更新版本，搜索索引使用重新验证的请求。
