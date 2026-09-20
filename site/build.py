"""Build the public reading site from published Markdown; no client framework required."""
from pathlib import Path
from urllib.parse import quote
from html.parser import HTMLParser
import html
import hashlib
import json
import re
import shutil
import markdown
from publishing import history, evidence_panel, metadata, feeds, PAGES, verification_meta

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / '_site'
BASE = '/llm-inference-notes/'
ORIGIN = 'https://akiyangdev.github.io'
REPO = 'https://github.com/AkiYangDev/llm-inference-notes'
TOPICS = {'sglang': 'SGLang', 'ascend': 'Ascend', 'fundamentals': '推理基础', 'distributed': '分布式推理', 'speculative-decoding': '投机推理', 'performance': '性能分析'}
DESCRIPTION = 'AkiYang 的大模型推理工程文档：SGLang、Ascend、源码与性能分析。'


def esc(value):
    return html.escape(str(value), quote=True)


def title_html(title):
    text = esc(title)
    for term in ('Chat Completion', 'DeepSeek-V4', 'Ascend NPU', 'KV Cache'):
        text = text.replace(term, f'<span class="keep">{term}</span>')
    return text


def shell(title, body, kind='home', route='', description=DESCRIPTION, dates=None):
    version = hashlib.sha256((ROOT / 'site/assets/style.css').read_bytes() + (ROOT / 'site/assets/app.js').read_bytes()).hexdigest()[:12]
    return f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(title)} · AkiYang</title><meta name="description" content="{esc(description)}">
{metadata(title, description, kind, route, ORIGIN, BASE, dates)}
{verification_meta(ROOT)}
<link rel="canonical" href="{ORIGIN}{BASE}{route}"><meta name="color-scheme" content="light dark">
<link rel="icon" href="{BASE}assets/favicon.svg" type="image/svg+xml">
<script>try{{const t=localStorage.getItem('aki-theme');document.documentElement.dataset.theme=t||(matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light')}}catch(e){{}}</script>
<link rel="stylesheet" href="{BASE}assets/style.css?v={version}">
<script type="module" src="{BASE}assets/app.js?v={version}"></script></head><body class="{kind}" id="top">
<a class="skip" href="#main">跳至正文</a><header class="header"><div class="header-inner">
<a class="brand" href="{BASE}" aria-label="AkiYang 首页"><span class="brand-icon" aria-hidden="true">A<span>.</span></span><span>AkiYang<span class="brand-caption">推理工程手记</span></span></a>
<nav aria-label="主导航"><a class="nav-articles" href="{BASE}articles/"{' aria-current="page"' if kind == 'archive-page' else ''}>文章</a><a class="nav-series" href="{BASE}series/">阅读地图</a><a class="nav-skills" href="{BASE}skills/"{' aria-current="page"' if 'skill' in kind else ''}>Skills</a><a class="nav-github" href="{REPO}" target="_blank" rel="noopener noreferrer">GitHub <span aria-hidden="true">↗</span></a><button class="search-trigger" type="button" aria-label="搜索文章与 Skills"><svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="10.5" cy="10.5" r="6.5"/><path d="m16 16 4 4"/></svg><span>搜索</span><kbd>/</kbd></button><button class="theme-toggle icon-button" aria-label="切换深色模式" title="切换深色模式"><svg class="moon" viewBox="0 0 24 24" aria-hidden="true"><path d="M20.5 13A8.5 8.5 0 0 1 11 3.5 8.5 8.5 0 1 0 20.5 13Z"/></svg><svg class="sun" viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="4"/><path d="M12 2v2m0 16v2M2 12h2m16 0h2M5 5l1.5 1.5m11 11L19 19M5 19l1.5-1.5m11-11L19 5"/></svg></button></nav></div></header>
{body}
<footer class="footer wrap"><a href="{BASE}" class="footer-brand">AkiYang<span>推理工程手记</span></a><div><a href="{BASE}feed.xml">RSS 订阅</a><a href="{BASE}sitemap.xml">站点地图</a><a href="{BASE}credits/">插画来源</a><a href="{REPO}">源代码 ↗</a><span>独立个人站 · 非 DeepSeek 官方</span></div></footer>
<dialog id="search-dialog" aria-labelledby="search-title"><div class="dialog-top"><h2 id="search-title"><span class="mascot mascot-search" aria-hidden="true"></span>搜索文章与 Skills</h2><button class="close-search icon-button" aria-label="关闭搜索">×</button></div><label for="search-input" class="sr-only">搜索标题、章节或正文</label><input id="search-input" type="search" placeholder="搜索标题、章节或正文…" autocomplete="off"><div id="search-results" aria-live="polite"></div><p class="dialog-hint">↑ ↓ 选择 · Enter 打开 · Esc 关闭</p></dialog>
<dialog id="diagram-dialog" aria-labelledby="diagram-title"><div class="dialog-top"><h2 id="diagram-title">工程图</h2><div class="diagram-controls"><button class="zoom-out icon-button" aria-label="缩小工程图">−</button><button class="zoom-reset" aria-label="工程图适应窗口">适应窗口</button><button class="zoom-in icon-button" aria-label="放大工程图">+</button><button class="close-diagram icon-button" aria-label="关闭工程图">×</button></div></div><div id="diagram-view" tabindex="0" aria-label="工程图，可滚动查看"></div></dialog>
<div class="toast" role="status" aria-live="polite"></div></body></html>'''


class SearchSections(HTMLParser):
    """Index human-readable content by heading, omitting diagram source."""
    def __init__(self):
        super().__init__()
        self.sections = [{'heading': '文章导读', 'anchor': '', 'parts': []}]
        self.heading = False
        self.skip = 0
    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'pre' and 'mermaid' in attrs.get('class', ''):
            self.skip += 1
        if tag in ('h2', 'h3'):
            self.sections.append({'heading': '', 'anchor': attrs.get('id', ''), 'parts': []})
            self.heading = True
    def handle_endtag(self, tag):
        if tag == 'pre' and self.skip:
            self.skip -= 1
        if tag in ('h2', 'h3'):
            self.heading = False
    def handle_data(self, data):
        if self.skip:
            return
        if self.heading:
            self.sections[-1]['heading'] += data
        else:
            self.sections[-1]['parts'].append(data)
    def result(self):
        return [{'heading': s['heading'], 'anchor': s['anchor'], 'text': re.sub(r'\s+', ' ', ' '.join(s['parts'])).strip()} for s in self.sections]


def tag_slug(tag):
    # ASCII route independent of label punctuation and translation.
    return hashlib.sha256(tag.encode('utf-8')).hexdigest()[:12]


def tag_links(tags):
    return ''.join(f'<a class="content-tag" href="{BASE}tags/{tag_slug(tag)}/">{esc(tag)}</a>' for tag in tags)


def card(a, number=1):
    return f'''<article class="article-card"><div class="card-index">{number:02d}<span> / ARTICLE</span></div><div class="card-content"><div class="card-meta"><span class="card-category">{esc(a['category'])}</span><span>约 {a['minutes']} 分钟阅读</span></div><a class="card-main" href="{a['url']}"><h3>{title_html(a['title'])}</h3><p>{esc(a['excerpt'])}</p><div class="card-bottom"><span>阅读全文</span><span aria-hidden="true">↗</span></div></a><nav class="article-tags" aria-label="文章标签">{tag_links(a['tags'])}</nav></div></article>'''


def render_document(path, routes):
    source = path.read_text(encoding="utf-8")
    md = markdown.Markdown(extensions=['fenced_code', 'tables', 'toc', 'sane_lists'], extension_configs={'toc': {'permalink': False}})
    rendered = md.convert(re.sub(r'^# .+\n', '', source, count=1))
    rendered = re.sub(r'<pre><code class="language-mermaid">([\s\S]*?)</code></pre>', r'<div class="diagram"><pre class="mermaid">\1</pre></div>', rendered)
    rendered = re.sub(r'(<table>[\s\S]*?</table>)', r'<div class="table-scroll" tabindex="0" role="region" aria-label="可横向滚动的表格">\1</div>', rendered)
    def link(match):
        attr, destination = match.group(1), html.unescape(match.group(2))
        if destination.startswith(('http:', 'https:', '#', 'mailto:', 'data:', '/')):
            return match.group(0)
        file_part, _, fragment = destination.partition('#')
        target = (path.parent / file_part).resolve()
        dest = target.relative_to(ROOT).as_posix()
        if attr == 'href' and target in routes:
            resolved = routes[target]
        else:
            prefix = REPO + '/blob/main/' if attr == 'href' else 'https://raw.githubusercontent.com/AkiYangDev/llm-inference-notes/main/'
            resolved = prefix + quote(dest)
        return attr + '="' + esc(resolved + ('#' + fragment if fragment else '')) + '"'
    rendered = re.sub(r'(href|src)="([^"]+)"', link, rendered)
    rendered = re.sub(r'<a href="(https://github.com/[^" ]+/blob/[a-f0-9]{40}/[^" ]+)"', r'<a class="source-reference" title="跳转到文章引用的固定版本源码" href="\1"', rendered)
    return md, rendered


def skill_card(skill):
    return f'''<a class="skill-card" href="{skill['url']}"><div class="skill-card-top"><span class="tag">Agent Skill</span><span aria-hidden="true">↗</span></div><h3>{esc(skill['title'])}</h3><p>{esc(skill['excerpt'])}</p><span class="skill-card-link">介绍与使用 <span aria-hidden="true">→</span></span></a>'''


def build_skills(paths, routes):
    skills = []
    for path in paths:
        source = path.read_text(encoding='utf-8')
        heading = re.search(r'^# (.+)$', source, re.M)
        if not heading:
            raise ValueError(f'Skill README needs a title: {path}')
        title = heading.group(1)
        md, rendered = render_document(path, routes)
        parser = SearchSections()
        parser.feed(rendered)
        sections = parser.result()
        sections[0]['heading'] = 'Skill 介绍'
        first_paragraph = re.search(r'<p>([\s\S]*?)</p>', rendered)
        excerpt = html.unescape(re.sub(r'<[^>]+>', '', first_paragraph.group(1))) if first_paragraph else title
        excerpt = excerpt[:160]
        slug = path.parent.name
        url = routes[path.resolve()]
        skill = {'title': title, 'topic': 'Skills', 'kind': 'skill', 'url': url, 'excerpt': excerpt, 'sections': sections}
        skills.append(skill)
        source_dir = REPO + '/tree/main/' + path.parent.relative_to(ROOT).as_posix()
        rules_url = REPO + '/blob/main/' + (path.parent / 'SKILL.md').relative_to(ROOT).as_posix()
        body = f'''<div class="reading-progress" aria-hidden="true"></div><main id="main" class="article-layout wrap"><div class="article-column"><nav class="breadcrumbs" aria-label="当前位置"><a href="{BASE}skills/">Skills</a><span>/</span><span>介绍与使用</span></nav><header class="article-header skill-header"><p class="eyebrow">AGENT SKILL</p><h1>{esc(title)}</h1><div class="article-meta"><span>AI Infra · 工程方法</span><a href="{source_dir}" target="_blank" rel="noopener noreferrer">GitHub 目录 ↗</a><a href="{rules_url}" target="_blank" rel="noopener noreferrer">查看规则 ↗</a><button class="copy-link">复制链接</button></div></header><details class="mobile-toc"><summary>本页目录 <span aria-hidden="true">⌄</span></summary>{md.toc}</details><article class="prose">{rendered}</article><div class="article-end"><a href="{BASE}skills/" class="text-link">← 全部 Skills</a><a href="{source_dir}" class="text-link" target="_blank" rel="noopener noreferrer">在 GitHub 查看文件 ↗</a></div></div><aside class="toc-panel" aria-label="章节导航"><span class="toc-label">本页目录</span>{md.toc}<a class="back-top" href="#top">↑ 回到顶部</a></aside></main>'''
        dest = OUT / 'skills' / slug / 'index.html'
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(shell(title, body, 'article-page skill-page', f'skills/{slug}/', excerpt), encoding='utf-8')
    intro = '可复用的工程方法，面向源码研究、技术写作与内容审核。'
    root_readme = ROOT / 'skills/README.md'
    if root_readme.exists():
        _, root_html = render_document(root_readme, routes)
        paragraph = re.search(r'<p>([\s\S]*?)</p>', root_html)
        if paragraph:
            intro = html.unescape(re.sub(r'<[^>]+>', '', paragraph.group(1)))
    cards = ''.join(skill_card(skill) for skill in skills)
    page = f'''<main id="main" class="archive skills-index wrap"><a class="back-link" href="{BASE}">← 首页</a><div class="archive-heading"><div><p class="eyebrow">METHODS &amp; PRACTICE</p><h1>Skills<span class="count">{len(skills):02d}</span></h1><p>{esc(intro)}</p></div><button class="archive-search">搜索文章与 Skills <span aria-hidden="true">↗</span></button></div><div class="skills-grid">{cards or '<p>暂无已发布的 Skill 介绍。</p>'}</div></main>'''
    (OUT / 'skills').mkdir(exist_ok=True)
    (OUT / 'skills/index.html').write_text(shell('Skills', page, 'skills-index-page', 'skills/', intro), encoding='utf-8')
    return skills


def published_series(paths, routes):
    """Only existing, titled articles participate in order and navigation."""
    available = {p.relative_to(ROOT).as_posix(): p for p in paths}
    series = []
    seen = set()
    for spec in json.loads((ROOT / 'site/series.json').read_text(encoding='utf-8')):
        items = []
        for entry in spec['items']:
            if entry['path'] in seen:
                raise ValueError('Duplicate series article: ' + entry['path'])
            seen.add(entry['path'])
            path = available.get(entry['path'])
            if path is None:
                continue
            heading = re.search(r'^# (.+)$', path.read_text(encoding='utf-8'), re.M)
            if heading:
                items.append({**entry, 'title': heading.group(1), 'url': routes[path.resolve()]})
        if items:
            series.append({**spec, 'items': items, 'url': BASE + 'series/#' + spec['id']})
    return series


def series_navigation(series, current=None, overview=False):
    if not series:
        return ''
    items = series['items']
    current_index = next((i for i, item in enumerate(items) if item['path'] == current), None)
    count = f'{current_index + 1:02d} / {len(items):02d}' if current_index is not None else f'{len(items)} 篇'
    links = ''.join(f'<li><a href="{item["url"]}"' + (' aria-current="page"' if item['path'] == current else '') + f' title="{esc(item["title"])}"><span class="series-number">{i+1:02d}</span><span>{esc(item["label"])}</span></a></li>' for i, item in enumerate(items))
    identity = f' id="{series["id"]}"' if overview else ''
    heading = f'<h2>{esc(series["title"])}</h2>' if overview else f'<a href="{series["url"]}">{esc(series["title"])}</a>'
    return f'<nav class="series-nav"{identity} aria-label="{esc(series["title"])}"><div class="series-heading">{heading}<span>{count}</span></div><ol>{links}</ol></nav>'


def series_pager(series, current):
    if not series:
        return ''
    items = series['items']
    position = next(i for i, item in enumerate(items) if item['path'] == current)
    links = []
    for offset, label, arrow, relation in [(-1, '上一篇', '←', 'prev'), (1, '下一篇', '→', 'next')]:
        index = position + offset
        if 0 <= index < len(items):
            item = items[index]
            links.append(f'<a class="series-page-link {relation}" rel="{relation}" href="{item["url"]}"><span class="pager-label">{arrow} {label}</span><strong>{esc(item["label"])}</strong><span class="pager-title">{esc(item["title"])}</span></a>')
    return '<nav class="series-pager" aria-label="系列前后篇">' + ''.join(links) + '</nav>' if links else ''


def reading_map(series, articles):
    lookup = {a['url']: a for a in articles}
    blocks = []
    for group in series:
        nodes = []
        total = sum(lookup[item['url']]['minutes'] for item in group['items'])
        for i, item in enumerate(group['items']):
            a = lookup[item['url']]
            nodes.append(f'''<li class="map-step" data-reading-item="{item['url']}"><a href="{item['url']}"><span class="map-number">{i+1:02d}</span><h3>{esc(item['label'])}</h3><p>{esc(item.get('description', a['excerpt']))}</p><span class="map-meta">约 {a['minutes']} 分钟 <span data-reading-status>未读</span></span></a></li>''')
        blocks.append(f'''<section class="reading-map" id="{group['id']}" data-reading-group><div class="map-heading"><div><p class="eyebrow">SOURCE READING PATH</p><h2>{esc(group['title'])}</h2><p>{len(group['items'])} 篇 · 约 {total} 分钟 · 推荐阅读顺序</p></div><a class="text-link" data-continue-reading href="{group['items'][0]['url']}">开始阅读 →</a></div><ol class="map-steps">{''.join(nodes)}</ol><div class="map-progress" role="progressbar" aria-label="系列阅读进度" aria-valuemin="0" aria-valuemax="{len(group['items'])}" aria-valuenow="0"><div class="map-progress-fill"></div></div><div class="map-footer"><span data-reading-summary aria-live="polite">已读 0 / {len(group['items'])} 篇</span><span>进度仅保存在当前浏览器</span></div></section>''')
    return ''.join(blocks)


def build():
    PAGES.clear()
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / 'assets').mkdir(parents=True)
    for path in (ROOT / 'site/assets').iterdir():
        if path.suffix != '.md':
            shutil.copy2(path, OUT / 'assets' / path.name)
    editorial = json.loads((ROOT / "site/articles.json").read_text(encoding="utf-8"))
    articles = []
    paths = [p for p in sorted((ROOT / 'docs').rglob('*.md')) if p.name != 'README.md']
    routes = {p.resolve(): BASE + 'articles/' + p.parent.name + '/' + p.stem + '/' for p in paths}
    skill_paths = [p for p in sorted((ROOT / 'skills').glob('*/README.md')) if (p.parent / 'SKILL.md').is_file()]
    routes.update({p.resolve(): BASE + 'skills/' + p.parent.name + '/' for p in skill_paths})
    routes[(ROOT / 'skills/README.md').resolve()] = BASE + 'skills/'
    skills = build_skills(skill_paths, routes)
    series_list = published_series(paths, routes)
    series_by_path = {item['path']: series for series in series_list for item in series['items']}
    order = {item['path']: i for series in series_list for i, item in enumerate(series['items'])}
    paths.sort(key=lambda p: (p.parent.name, order.get(p.relative_to(ROOT).as_posix(), 10000), p.name))
    for path in paths:
        source = path.read_text(encoding='utf-8')
        heading = re.search(r'^# (.+)$', source, re.M)
        if not heading:
            continue
        title = heading.group(1)
        topic_id = path.parent.name
        topic = TOPICS.get(topic_id, topic_id)
        slug = topic_id + '/' + path.stem
        url = routes[path.resolve()]
        minutes = max(1, round(len(re.sub(r'```[\s\S]*?```', '', source)) / 650))
        md, rendered = render_document(path, routes)
        parser = SearchSections()
        parser.feed(rendered)
        sections = parser.result()
        entry = editorial.get(path.relative_to(ROOT).as_posix(), {})
        excerpt = entry.get('summary') or sections[0]['text'][:110]
        a = {'title': title, 'topic': topic, 'topic_id': topic_id, 'url': url, 'minutes': minutes, 'excerpt': excerpt, 'sections': sections}
        a['category'] = entry.get('category', {'ascend': '部署实践', 'fundamentals': '基础原理'}.get(topic_id, '推理工程'))
        a['tags'] = entry.get('tags', [topic])
        if not isinstance(a['tags'], list) or not a['tags'] or any(not isinstance(t, str) or not t.strip() for t in a['tags']) or len(set(a['tags'])) != len(a['tags']):
            raise ValueError(f'Invalid tags for {path}')
        articles.append(a)
        current_path = path.relative_to(ROOT).as_posix()
        dates = history(ROOT, path)
        a.update(dates)
        publication = evidence_panel(source, dates, REPO, current_path)
        current_series = series_by_path.get(current_path)
        series_nav = series_navigation(current_series, current_path)
        pager = series_pager(current_series, current_path)
        if current_series:
            pager = f'<div class="reading-actions"><span class="mascot mascot-complete" aria-hidden="true"></span><button class="reading-complete" data-article-url="{url}" aria-pressed="false">标记为已读</button><a class="text-link" href="{BASE}series/">返回阅读地图 →</a></div>' + pager
        title_parts = title.split('：', 1)
        display_title = title_html(title_parts[0]) + (f'<span class="title-sub">{title_html(title_parts[1])}</span>' if len(title_parts) == 2 else '')
        body = f'''<div class="reading-progress" aria-hidden="true"></div><main id="main" class="article-layout wrap"><div class="article-column"><nav class="breadcrumbs" aria-label="当前位置"><a href="{BASE}articles/">文章</a><span>/</span><a href="{BASE}topics/{topic_id}/">{esc(topic)}</a></nav><header class="article-header"><h1>{display_title}</h1><div class="article-meta"><span>AkiYang</span><span>约 {minutes} 分钟阅读</span><a href="{REPO}/blob/main/{path.relative_to(ROOT).as_posix()}" target="_blank" rel="noopener noreferrer">阅读源码文档 ↗</a><button class="copy-link">复制链接</button></div><nav class="article-tags header-tags" aria-label="文章标签">{tag_links(a["tags"])}</nav>{publication}<div class="resume-reading" hidden><button class="resume-position">继续上次阅读</button><button class="forget-position">清除位置</button><span>位置仅保存在当前浏览器</span></div></header>{series_nav}<details class="mobile-toc"><summary>本页目录 <span aria-hidden="true">⌄</span></summary>{md.toc}</details><article class="prose">{rendered}</article>{pager}<div class="article-end"><div><span class="eyebrow">读到这里</span><p>从一条请求，看见整个系统。</p></div><a href="{BASE}articles/" class="text-link">返回文章目录 <span aria-hidden="true">↗</span></a></div></div><aside class="toc-panel" aria-label="章节导航"><span class="toc-label">本页目录</span>{md.toc}<a class="back-top" href="#top">↑ 回到顶部</a></aside></main>'''
        output = OUT / 'articles' / slug / 'index.html'
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(shell(title, body, 'article-page', 'articles/' + slug + '/', excerpt, dates), encoding='utf-8')
    topics = sorted({a['topic_id'] for a in articles})
    topic_links = ''.join(f'<a class="topic-link" href="{BASE}topics/{t}/">{esc(TOPICS.get(t,t))}<span>{sum(a["topic_id"] == t for a in articles):02d}</span></a>' for t in topics)
    featured_url = articles[0]['url'] if articles else BASE + 'articles/'
    latest = sorted(articles, key=lambda a: (a.get('published', ''), a['url']), reverse=True)[:3]
    cards = ''.join(card(a, i + 1) for i, a in enumerate(latest))
    recommended = ''.join(f'<a class="recommend-link" href="{a["url"]}"><span>{label}</span><strong>{title_html(a["title"])}</strong><span aria-hidden="true">→</span></a>' for a, label in [(articles[0], '从这里开始'), (next(a for a in articles if 'distributed-parallel' in a['url']), '理解并行')])
    knowledge_map = reading_map(series_list, articles)
    body = f'''<main id="main"><section class="hero wrap" aria-labelledby="hero-title"><div class="hero-copy"><p class="eyebrow"><span class="tiny-line" aria-hidden="true"></span> AI INFRA NOTES</p><h1 id="hero-title">把推理原理，<br><span>写成读得懂的手记。</span></h1><p class="hero-description">DeepSeek · SGLang · Ascend<br>记录推理系统的原理、源码与工程实践。</p><div class="hero-actions"><a class="button-primary" href="{featured_url}">开始阅读 <span aria-hidden="true">↗</span></a><a class="text-link hero-secondary" href="{BASE}series/">阅读地图 <span aria-hidden="true">→</span></a></div><div class="hero-topics"><span>SGLang</span><span>Ascend NPU</span><span>LLM Inference</span></div></div><div class="hero-art"><div class="art-orbit" aria-hidden="true"></div><img src="{BASE}assets/whale-editor.webp" alt="拿着笔记本和钢笔的蓝发鲸鱼娘" width="937" height="1678" fetchpriority="high"><span class="art-caption">SOURCE / NOTES / IDEAS</span></div></section><section id="articles" class="articles-section wrap"><div class="section-heading"><div><p class="eyebrow">THE JOURNAL</p><h2>最新手记</h2></div><a href="{BASE}articles/" class="text-link">浏览全部 <span aria-hidden="true">↗</span></a></div><div class="article-list home-article-list">{cards}</div><div class="topic-strip"><span>按专题阅读</span>{topic_links}</div></section><section class="reading-entry wrap" aria-label="阅读路线"><span class="mascot mascot-reading" aria-hidden="true"></span><div><p class="eyebrow">READING SERIES</p><h2>沿着源码，逐步深入</h2><p>请求执行 · 模型内部 · 分布式并行 · 投机推理</p></div><a class="text-link" href="{BASE}series/">查看完整阅读地图 →</a></section><section class="home-recommended wrap" aria-labelledby="recommended-title"><div class="section-heading"><div><p class="eyebrow">EDITOR’S PICKS</p><h2 id="recommended-title">推荐阅读</h2></div></div><div class="recommend-list">{recommended}</div></section></main>'''
    if skills:
        skill_section = f'''<section class="home-skills wrap" aria-labelledby="home-skills-title"><div class="section-heading"><div><p class="eyebrow">METHODS &amp; PRACTICE</p><h2 id="home-skills-title">Skills <span class="count">{len(skills):02d}</span></h2></div><a class="text-link" href="{BASE}skills/">浏览 Skills <span aria-hidden="true">↗</span></a></div><div class="skills-grid">{''.join(skill_card(skill) for skill in skills[:3])}</div></section>'''
        body = body.replace('</main>', skill_section + '</main>')
    series_page = f'''<main id="main" class="archive wrap"><a class="back-link" href="{BASE}">← 首页</a><div class="archive-heading"><div><p class="eyebrow">THE READING ATLAS</p><h1>源码阅读地图</h1><p>从整体请求到模型内部，沿着已发布的文章深入系统。</p></div></div>{knowledge_map}<p class="map-note">这里展示文章的推荐阅读顺序，并非运行时调用图。读完文章后可在文末标记已读。</p></main>'''
    (OUT / 'series').mkdir(exist_ok=True)
    (OUT / 'series/index.html').write_text(shell('源码阅读地图', series_page, 'series-index-page', 'series/'), encoding='utf-8')
    (OUT / 'index.html').write_text(shell('推理工程手记', body), encoding='utf-8')
    all_tags = sorted({tag for a in articles for tag in a['tags']}, key=str.casefold)
    archive_routes = [(None, None)] + [(t, None) for t in topics] + [(None, tag) for tag in all_tags]
    for topic_id, active_tag in archive_routes:
        selected = [a for a in articles if (topic_id is None or a['topic_id'] == topic_id) and (active_tag is None or active_tag in a['tags'])]
        label = TOPICS.get(topic_id, topic_id) if topic_id else '文章目录'
        route = f'topics/{topic_id}/' if topic_id else 'articles/'
        if active_tag:
            label = active_tag
            route = f'tags/{tag_slug(active_tag)}/'
        filters = f'<a href="{BASE}articles/"' + (' aria-current="page"' if not topic_id and not active_tag else '') + '>全部文章</a>'
        filters += ''.join(f'<a href="{BASE}topics/{t}/"' + (' aria-current="page"' if t == topic_id else '') + f'>{esc(TOPICS.get(t,t))}</a>' for t in topics)
        tag_filters = ''.join(f'<a href="{BASE}tags/{tag_slug(t)}/"' + (' aria-current="page"' if active_tag == t else '') + f'>{esc(t)}<span>{sum(t in a["tags"] for a in articles)}</span></a>' for t in all_tags)
        series_overview = ''.join(series_navigation(series, overview=True) for series in series_list if not active_tag and (topic_id is None or series['topic'] == topic_id))
        archive_cards = ''.join(f'<section class="journal-category"><h2 class="category-label">{esc(category)}</h2>' + ''.join(card(a, i + 1) for i, a in enumerate(selected) if a['category'] == category) + '</section>' for category in dict.fromkeys(a['category'] for a in selected))
        page = f'''<main id="main" class="archive wrap"><a class="back-link" href="{BASE}">← 首页</a><div class="archive-heading"><div><p class="eyebrow">INFERENCE ARCHIVE</p><h1>{esc(label)}<span class="count">{len(selected):02d}</span></h1><p>关于推理系统的原理、源码与工程实践。</p></div><button class="archive-search">搜索文章 <span aria-hidden="true">↗</span></button></div><nav class="topic-filters" aria-label="筛选专题">{filters}</nav><div class="tag-browser"><span class="filter-label">按内容标签浏览</span><nav class="tag-filters" aria-label="筛选内容标签">{tag_filters}</nav></div>{series_overview}<div class="archive-list">{archive_cards}</div></main>'''
        dest = OUT / route / 'index.html'
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(shell(label, page, 'archive-page', route, f'{label}相关文章：源码分析、执行链路与推理工程实践。'), encoding='utf-8')
    (OUT / 'search.json').write_text(json.dumps(articles + skills, ensure_ascii=False), encoding='utf-8')
    (OUT / '.nojekyll').touch()
    credits = f'''<main id="main" class="credits wrap"><a class="back-link" href="{BASE}">← 首页</a><h1>插画来源</h1><p>本站为 AkiYang 的独立个人技术站，非 DeepSeek 官方网站。</p><h2>角色设计</h2><p>根据站点所有者提供的作者信息，鲸鱼娘角色设计署名 <a href="https://space.bilibili.com/4168597/dynamic">ZipZipPipe</a>。角色设计署名不等于下列每张衍生插画都由该作者绘制。</p><h2>首页立绘</h2><p>站点所有者提供持笔记本与钢笔的鲸鱼娘素材，本站使用 AI 辅助制作透明背景展示版本，可能存在局部细节变化。该张衍生插画的具体绘者未另行核实。</p><h2>Q 版状态素材</h2><p>阅读、搜索、完成与迷路四种状态为 AI 生成的角色延展素材，非 ZipZipPipe 原作。不对原角色或用户提供的插画主张原创或再许可。</p><p class="credits-note">历史素材与处理记录见 <a href="{REPO}/blob/main/site/assets/SOURCES.md">SOURCES.md</a>。</p></main>'''
    (OUT / 'credits').mkdir()
    (OUT / 'credits/index.html').write_text(shell('插画来源', credits, 'credits-page', 'credits/'), encoding='utf-8')
    error = f'<main id="main" class="error-page wrap"><span class="mascot mascot-lost" aria-hidden="true"></span><span class="eyebrow">404 / PAGE NOT FOUND</span><h1>这一页，游到别处去了。</h1><p>链接可能已经变更，可以从文章目录继续阅读。</p><a class="button-primary" href="{BASE}articles/">浏览文章 →</a></main>'
    (OUT / '404.html').write_text(shell('页面未找到', error, route='404.html'), encoding='utf-8')
    feeds(OUT, articles, ORIGIN, BASE)
    print(f'Built {len(articles)} article(s), {len(topics)} topic(s), {len(skills)} skill(s) into {OUT}')

if __name__ == '__main__':
    build()

