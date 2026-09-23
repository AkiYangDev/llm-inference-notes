"""Build the public reading site from published Markdown; no client framework required."""
from pathlib import Path
from urllib.parse import quote
from html.parser import HTMLParser
import html
import base64
import hashlib
import json
import re
import shutil
import markdown
from math_blocks import MathBlocksExtension
from publishing import history, evidence_panel, metadata, feeds, PAGES, verification_meta

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / '_site'
BASE = '/llm-inference-notes/'
ORIGIN = 'https://akiyangdev.github.io'
REPO = 'https://github.com/AkiYangDev/llm-inference-notes'
TOPICS = {'sglang': 'SGLang', 'pr-reviews': 'PR 精读', 'ascend': 'Ascend', 'fundamentals': '推理基础', 'distributed': '分布式推理', 'speculative-decoding': '投机解码', 'performance': '性能分析'}
CONTENT_ROLES = {
    'entry': {'label': '入门', 'description': '从具体问题开始，理解推理系统的基础概念，找到适合自己的第一篇文章。'},
    'advanced': {'label': '进阶', 'description': '从概念走向实现，适合已经掌握基础概念后继续进入 SGLang 与 DeepSeek 工程链路。'},
    'deep': {'label': '深入研究', 'description': '源码、性能、投机解码与 PR 级深挖，面向正在做推理工程的人。'},
    'reference': {'label': '速查', 'description': '可反复查阅的名词表、并行拓扑、通信语义与知识地图。'},
}
DESCRIPTION = 'AkiYang 的大模型推理工程文档：SGLang、Ascend、源码与性能分析。'


def esc(value):
    return html.escape(str(value), quote=True)


def title_html(title):
    text = esc(title)
    for term in ('Chat Completion', 'DeepSeek-V4', 'Ascend NPU', 'KV Cache'):
        text = text.replace(term, f'<span class="keep">{term}</span>')
    return text


def shell(title, body, kind='home', route='', description=DESCRIPTION, dates=None, browser_title=None, meta_description=None, indexable=True):
    version = hashlib.sha256((ROOT / 'site/assets/style.css').read_bytes() + (ROOT / 'site/assets/app.js').read_bytes() + (ROOT / 'site/assets/search.mjs').read_bytes()).hexdigest()[:12]
    browser_title = browser_title or title
    meta_description = meta_description or description
    return f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(browser_title)} · AkiYang</title><meta name="description" content="{esc(meta_description)}">
{metadata(title, meta_description, kind, route, ORIGIN, BASE, dates, indexable=indexable)}
{verification_meta(ROOT)}
<link rel="canonical" href="{ORIGIN}{BASE}{route}"><meta name="color-scheme" content="light dark">
<link rel="icon" href="{BASE}assets/favicon.svg" type="image/svg+xml">
<script>try{{const t=localStorage.getItem('aki-theme');document.documentElement.dataset.theme=t||(matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light')}}catch(e){{}}</script>
<link rel="stylesheet" href="{BASE}assets/style.css?v={version}">
<script type="module" src="{BASE}assets/app.js?v={version}"></script></head><body class="{kind}" id="top">
<a class="skip" href="#main">跳至正文</a><header class="header"><div class="header-inner">
<a class="brand" href="{BASE}" aria-label="AkiYang 首页"><span class="brand-icon" aria-hidden="true">A<span>.</span></span><span>AkiYang<span class="brand-caption">推理工程手记</span></span></a>
<nav aria-label="主导航"><a class="nav-home" href="{BASE}"{' aria-current="page"' if kind == 'home' else ''}>首页</a><a class="nav-articles" href="{BASE}articles/"{' aria-current="page"' if kind == 'archive-page' else ''}>文章</a><a class="nav-series" href="{BASE}series/"{' aria-current="page"' if kind == 'series-index-page' else ''}>阅读地图</a><a class="nav-skills" href="{BASE}skills/"{' aria-current="page"' if 'skill' in kind else ''}>Skills</a><a class="nav-github" href="{REPO}" target="_blank" rel="noopener noreferrer">GitHub <span aria-hidden="true">↗</span></a><button class="search-trigger" type="button" aria-label="搜索文章与 Skills"><svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="10.5" cy="10.5" r="6.5"/><path d="m16 16 4 4"/></svg><span>搜索</span><kbd>/</kbd></button><button class="theme-toggle icon-button" aria-label="切换深色模式" title="切换深色模式"><svg class="moon" viewBox="0 0 24 24" aria-hidden="true"><path d="M20.5 13A8.5 8.5 0 0 1 11 3.5 8.5 8.5 0 1 0 20.5 13Z"/></svg><svg class="sun" viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="4"/><path d="M12 2v2m0 16v2M2 12h2m16 0h2M5 5l1.5 1.5m11 11L19 19M5 19l1.5-1.5m11-11L19 5"/></svg></button><button class="menu-toggle" aria-label="打开导航菜单" aria-controls="menu-dialog" hidden>菜单</button></nav></div></header>
{body}
<footer class="footer wrap"><a href="{BASE}" class="footer-brand">AkiYang<span>推理工程手记</span></a><div><a href="{BASE}articles/?saved=1">稍后阅读</a><a href="{BASE}feed.xml">RSS 订阅</a><a href="{BASE}sitemap.xml">站点地图</a><a href="{BASE}credits/">插画来源</a><a href="{REPO}">源代码 ↗</a><span>独立个人站 · 非 DeepSeek 官方</span></div></footer>
<dialog id="menu-dialog" aria-labelledby="menu-title"><div class="dialog-top"><h2 id="menu-title">探索手记</h2><button class="close-menu icon-button" aria-label="关闭导航菜单">×</button></div><nav aria-label="移动端导航"><a href="{BASE}">首页</a><a href="{BASE}articles/">全部文章</a><a href="{BASE}series/">阅读地图</a><a href="{BASE}articles/?saved=1">稍后阅读</a><a href="{BASE}skills/">Skills</a><a href="{BASE}feed.xml">RSS 订阅</a><a href="{REPO}" target="_blank" rel="noopener noreferrer">GitHub ↗</a></nav></dialog>
<dialog id="search-dialog" aria-labelledby="search-title"><div class="dialog-top"><h2 id="search-title"><span class="mascot mascot-search" aria-hidden="true"></span>搜索文章与 Skills</h2><button class="close-search icon-button" aria-label="关闭搜索">×</button></div><label for="search-input" class="sr-only">搜索标题、章节或正文</label><input id="search-input" type="search" placeholder="搜索标题、章节或正文…" autocomplete="off"><div class="search-suggestions" aria-label="常用搜索"><button data-search-query="SGLang">SGLang</button><button data-search-query="KV Cache">KV Cache</button><button data-search-query="DSpark">DSpark</button><button data-search-query="Ascend 910C">Ascend 910C</button></div><div id="search-results" aria-live="polite"></div><p class="dialog-hint">↑ ↓ 选择 · Enter 打开 · Esc 关闭 · Ctrl / ⌘ K 搜索</p></dialog>
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
    search_text = ' '.join([a['title'], a['excerpt'], a['topic'], *a['tags']])
    date = a.get('modified', '')
    stamp = f'<time datetime="{esc(date)}">更新于 {date[:10]}</time>' if date else ''
    return f'''<article class="article-card" data-url="{a['url']}" data-search="{esc(search_text)}" data-minutes="{a['minutes']}" data-modified="{esc(date)}"><div class="card-content"><div class="card-meta"><span class="card-category">{esc(a['role_label'])} / {esc(a['category'])}</span><span class="read-badge" hidden>已读</span>{bookmark_button(a['url'], a['title'])}</div><a class="card-main" href="{a['url']}"><h3>{title_html(a['title'])}</h3><p>{esc(a['excerpt'])}</p></a><nav class="article-tags" aria-label="文章标签">{tag_links(a['tags'])}</nav><div class="card-bottom"><span>{a['minutes']} 分钟阅读</span>{stamp}<a href="{a['url']}" aria-label="阅读全文：{esc(a['title'])}">阅读全文 <span aria-hidden="true">↗</span></a></div></div></article>'''


def bookmark_button(url, title):
    return f'''<button class="bookmark-button" type="button" data-bookmark="{url}" data-title="{esc(title)}" aria-pressed="false" aria-label="稍后阅读：{esc(title)}" title="稍后阅读" hidden><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 4h12v17l-6-4-6 4Z"/></svg><span>稍后读</span></button>'''


def reading_toolbar(url, title):
    return f'''<div class="reading-toolbar" role="group" aria-label="阅读工具" hidden>{bookmark_button(url, title)}<div class="font-controls"><span>字号</span><button class="font-smaller" aria-label="减小正文字号">A−</button><output class="font-value" aria-live="polite">18</output><button class="font-larger" aria-label="增大正文字号">A+</button></div><button class="focus-toggle" aria-pressed="false">专注阅读</button></div>'''


def related_articles(article, articles):
    def score(other):
        return 3 * (article['topic_id'] == other['topic_id']) + len(set(article['tags']) & set(other['tags']))
    candidates = sorted((a for a in articles if a['url'] != article['url'] and score(a)), key=score, reverse=True)[:3]
    if not candidates:
        return ''
    links = ''.join(f'<a href="{a["url"]}"><span>{esc(a["topic"])} · {a["minutes"]} 分钟</span><strong>{esc(a["title"])}</strong></a>' for a in candidates)
    return f'<section class="related-articles" aria-label="延伸阅读"><h2>接着读</h2><div>{links}</div></section>'


def render_document(path, routes):
    source = path.read_text(encoding="utf-8")
    md = markdown.Markdown(extensions=['fenced_code', 'tables', 'toc', 'sane_lists', MathBlocksExtension()], extension_configs={'toc': {'permalink': False}})
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

    # The high-quality homepage hero is stored as small text payload chunks so
    # repository tooling can transport it reliably. Rebuild one normal WebP for
    # the published site; browsers still request a single image.
    hero_payload_dir = ROOT / 'site/asset-payloads/whale-hero-v2'
    hero_payload = ''.join(
        part.read_text(encoding='ascii').strip()
        for part in sorted(hero_payload_dir.glob('part-*.b64'))
    )
    hero_bytes = base64.b64decode(hero_payload, validate=True)
    hero_sha256 = hashlib.sha256(hero_bytes).hexdigest()
    if len(hero_bytes) != 40634 or hero_sha256 != '0b04d52e34bd8192dfa28db1937a442432a9cd1f6b81a62a19ea2ce9cc99ffe0':
        raise RuntimeError('whale-hero-v2 payload is incomplete or corrupted')
    (OUT / 'assets' / 'whale-hero-v2.webp').write_bytes(hero_bytes)
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
        article_path = path.relative_to(ROOT).as_posix()
        entry = editorial.get(article_path)
        if entry is None:
            raise ValueError(f'Missing editorial metadata for {path}')
        topic_id = entry.get('topic', path.parent.name)
        if topic_id not in TOPICS:
            raise ValueError(f'Unknown topic {topic_id!r} for {path}')
        topic = TOPICS[topic_id]
        # Keep article URLs stable and tied to repository paths even when the
        # editorial topic differs from the physical docs directory.
        slug = path.parent.name + '/' + path.stem
        url = routes[path.resolve()]
        minutes = max(1, round(len(re.sub(r'(```|~~~)[\s\S]*?\1', '', source)) / 650))
        md, rendered = render_document(path, routes)
        parser = SearchSections()
        parser.feed(rendered)
        sections = parser.result()
        excerpt = entry.get('summary') or sections[0]['text'][:110]
        a = {'title': title, 'topic': topic, 'topic_id': topic_id, 'url': url, 'minutes': minutes, 'excerpt': excerpt, 'sections': sections}
        a['category'] = entry.get('category', {'ascend': '部署实践', 'fundamentals': '基础原理'}.get(topic_id, '推理工程'))
        role = entry.get('role')
        if role not in CONTENT_ROLES:
            raise ValueError(f'Missing or invalid content role for {path}: {role!r}')
        a['role'] = role
        a['role_label'] = CONTENT_ROLES[role]['label']
        a['tags'] = entry.get('tags', [topic])
        if not isinstance(a['tags'], list) or not a['tags'] or any(not isinstance(t, str) or not t.strip() for t in a['tags']) or len(set(a['tags'])) != len(a['tags']):
            raise ValueError(f'Invalid tags for {path}')
        articles.append(a)
        current_path = article_path
        dates = history(ROOT, path)
        a.update(dates)
        publication = evidence_panel(source, dates, REPO, current_path)
        current_series = series_by_path.get(current_path)
        series_nav = series_navigation(current_series, current_path)
        pager = series_pager(current_series, current_path)
        if current_series:
            series_nav = f'<details class="article-series"><summary>系列阅读 · {esc(current_series["title"])}<span>展开篇目</span></summary>{series_nav}</details>'
        pager = f'<div class="reading-actions"><span class="mascot mascot-complete" aria-hidden="true"></span><button class="reading-complete" data-article-url="{url}" aria-pressed="false">标记为已读</button><a class="text-link" href="{BASE}series/">返回阅读地图 →</a></div>' + pager
        title_parts = title.split('：', 1)
        display_title = title_html(title_parts[0]) + (f'<span class="title-sub">{title_html(title_parts[1])}</span>' if len(title_parts) == 2 else '')
        body = f'''<div class="reading-progress" aria-hidden="true"></div><main id="main" class="article-layout wrap"><div class="article-column"><nav class="breadcrumbs" aria-label="当前位置"><a href="{BASE}articles/">文章</a><span>/</span><a href="{BASE}topics/{topic_id}/">{esc(topic)}</a></nav><header class="article-header"><h1>{display_title}</h1><div class="article-meta"><span>AkiYang</span><span>约 {minutes} 分钟阅读</span><a href="{REPO}/blob/main/{path.relative_to(ROOT).as_posix()}" target="_blank" rel="noopener noreferrer">阅读源码文档 ↗</a><button class="copy-link">复制链接</button></div><nav class="article-tags header-tags" aria-label="文章标签"><a class="content-tag" href="{BASE}levels/{a['role']}/">{esc(a['role_label'])}</a>{tag_links(a["tags"])}</nav><details class="source-details"><summary>版本与源码依据</summary>{publication}</details><div class="resume-reading" hidden><button class="resume-position">继续上次阅读</button><button class="forget-position">清除位置</button><span>位置仅保存在当前浏览器</span></div></header>{reading_toolbar(url, title)}{series_nav}<details class="mobile-toc"><summary>本页目录 <span aria-hidden="true">⌄</span></summary>{md.toc}</details><article class="prose">{rendered}</article>{pager}<!--related-articles--><div class="article-end"><div><span class="eyebrow">读到这里</span><p>从一条请求，看见整个系统。</p></div><a href="{BASE}articles/" class="text-link">返回文章目录 <span aria-hidden="true">↗</span></a></div></div><aside class="toc-panel" aria-label="章节导航"><span class="toc-label">本页目录</span>{md.toc}<a class="back-top" href="#top">↑ 回到顶部</a></aside></main>'''
        output = OUT / 'articles' / slug / 'index.html'
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(shell(title, body, 'article-page', 'articles/' + slug + '/', excerpt, dates, browser_title=entry.get('seo_title'), meta_description=entry.get('seo_summary', excerpt)), encoding='utf-8')
    for a in articles:
        output = OUT / a['url'].removeprefix(BASE) / 'index.html'
        output.write_text(output.read_text(encoding='utf-8').replace('<!--related-articles-->', related_articles(a, articles)), encoding='utf-8')
    # Preserve legacy article URLs after information-architecture moves.
    # GitHub Pages cannot emit HTTP 301 responses, so old routes are static
    # noindex redirect pages with a canonical link to the new location.
    redirects_path = ROOT / 'site/redirects.json'
    redirects = json.loads(redirects_path.read_text(encoding='utf-8')) if redirects_path.exists() else {}
    for old_route, new_route in redirects.items():
        old_route = old_route.strip('/') + '/'
        new_route = new_route.strip('/') + '/'
        target = BASE + new_route
        canonical = ORIGIN + target
        dest = OUT / old_route / 'index.html'
        dest.parent.mkdir(parents=True, exist_ok=True)
        redirect_html = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="robots" content="noindex,follow"><meta name="description" content="文章已迁移到新的专题目录。"><link rel="canonical" href="{canonical}"><noscript><meta http-equiv="refresh" content="0; url={target}"></noscript><title>文章已迁移 · AkiYang</title><script>location.replace({json.dumps(target)} + location.search + location.hash);</script></head><body><main><p>文章已迁移到新的专题目录。</p><p><a href="{target}">前往新地址</a></p></main></body></html>'''
        dest.write_text(redirect_html, encoding='utf-8')

    topics = sorted({a['topic_id'] for a in articles})
    topic_links = ''.join(f'<a class="topic-link" href="{BASE}topics/{t}/">{esc(TOPICS.get(t,t))}<span>{sum(a["topic_id"] == t for a in articles):02d}</span></a>' for t in topics)
    role_links = ''.join(f'<a class="topic-link" href="{BASE}levels/{role}/">{esc(spec["label"])}<span>{sum(a["role"] == role for a in articles):02d}</span></a>' for role, spec in CONTENT_ROLES.items())
    # Keep the homepage entry point stable as new topic directories are added.
    # Using articles[0] made "开始阅读" depend on alphabetical path ordering;
    # adding docs/ascend therefore unexpectedly changed the beginner entry to W8A8.
    start_path = (ROOT / 'docs/distributed/why-large-models-need-multi-card-tp-dp-ep.md').resolve()
    start_url = routes.get(start_path)
    start_article = next((a for a in articles if a['url'] == start_url), articles[0] if articles else None)
    featured_url = start_article['url'] if start_article else BASE + 'articles/'
    latest = sorted(articles, key=lambda a: (a.get('published', ''), a['url']), reverse=True)[:3]
    cards = ''.join(card(a) for a in latest)
    knowledge_map = reading_map(series_list, articles)
    topic_tiles = ''.join(f'<a href="{BASE}topics/{t}/"><span>{esc(TOPICS[t])}</span><span>{sum(a["topic_id"] == t for a in articles):02d} 篇 <b aria-hidden="true">↗</b></span></a>' for t in TOPICS if t in topics)
    routes_preview = ''.join(f'<a class="route-preview" href="{group["url"]}"><span>{len(group["items"]):02d} 篇</span><strong>{esc(group["title"])}</strong><span aria-hidden="true">→</span></a>' for group in series_list[:3])
    body = f'''<main id="main">
<section class="hero wrap" aria-labelledby="hero-title"><div class="hero-copy"><p class="eyebrow">AI INFRA / ENGINEERING NOTES</p><h1 id="hero-title">大模型推理<span>工程手记。</span></h1><p class="hero-description">SGLang 源码、Ascend 910C 部署<br>与 DeepSeek 推理实践。</p><div class="hero-actions"><a class="button-primary" href="{featured_url}">从基础开始 <span aria-hidden="true">↗</span></a><a class="text-link" href="{BASE}articles/">浏览全部文章 →</a></div><div class="hero-stats"><span><strong>{len(articles):02d}</strong> 篇手记</span><span><strong>{len(topics):02d}</strong> 个专题</span><a href="{BASE}feed.xml">RSS 订阅 ↗</a></div></div><div class="hero-art"><img src="{BASE}assets/whale-hero-v2.webp" alt="蓝发鲸尾女仆与蓝白海洋科技场景" width="1000" height="450" fetchpriority="high" decoding="async"></div><span class="hero-caption" aria-hidden="true">FIELD NOTES · SOURCE &amp; SYSTEMS</span></section>
<div class="home-entry wrap"><a href="{BASE}levels/entry/"><span>01 / START</span><strong>第一次来，从基础开始</strong><span>Token、推理阶段与多卡并行 →</span></a><a href="{BASE}series/"><span>02 / EXPLORE</span><strong>沿着源码，串起系统</strong><span>请求、调度、模型与缓存 →</span></a><a href="{BASE}levels/reference/"><span>03 / REFERENCE</span><strong>遇到概念，随时查阅</strong><span>名词、通信与并行拓扑 →</span></a></div>
<div class="home-columns wrap"><section class="articles-section" aria-labelledby="latest-title"><div class="section-heading"><div><p class="eyebrow">THE JOURNAL</p><h2 id="latest-title">最新手记</h2></div><a href="{BASE}articles/" class="text-link">全部 {len(articles)} 篇 ↗</a></div><div class="article-list home-article-list">{cards}</div></section><aside class="home-sidebar" aria-label="阅读入口"><a class="last-reading" data-last-reading hidden><span>继续上次阅读</span><strong></strong><span>回到阅读位置 →</span></a><section class="sidebar-panel"><p class="eyebrow">BROWSE BY TOPIC</p><h2>按专题探索</h2><div class="topic-tiles">{topic_tiles}</div></section><section class="sidebar-panel atlas-teaser"><span class="mascot mascot-reading" aria-hidden="true"></span><p class="eyebrow">THE READING ATLAS</p><h2>把知识连成一条线</h2><p>按推荐顺序阅读已发布系列。</p>{routes_preview}<a class="text-link" href="{BASE}series/">完整阅读地图 →</a></section><a class="saved-entry" href="{BASE}articles/?saved=1">稍后阅读 <span>打开已收藏的文章 →</span></a></aside></div>
</main>'''
    if skills:
        skill_section = f'''<section class="home-skills wrap" aria-labelledby="home-skills-title"><div class="section-heading"><div><p class="eyebrow">TOOLS &amp; METHODS</p><h2 id="home-skills-title">可复用的工程方法</h2></div><a class="text-link" href="{BASE}skills/">全部 Skills ↗</a></div><div class="skills-grid">{''.join(skill_card(skill) for skill in skills)}</div></section>'''
        body = body.replace('</main>', skill_section + '</main>')
    series_page = f'''<main id="main" class="archive wrap"><a class="back-link" href="{BASE}">← 首页</a><div class="archive-heading"><div><p class="eyebrow">THE READING ATLAS</p><h1>源码阅读地图</h1><p>从整体请求到模型内部，沿着已发布的文章深入系统。</p></div></div>{knowledge_map}<p class="map-note">这里展示文章的推荐阅读顺序，并非运行时调用图。读完文章后可在文末标记已读。</p></main>'''
    (OUT / 'series').mkdir(exist_ok=True)
    (OUT / 'series/index.html').write_text(shell('源码阅读地图', series_page, 'series-index-page', 'series/'), encoding='utf-8')
    (OUT / 'index.html').write_text(shell('推理工程手记', body), encoding='utf-8')
    all_tags = sorted({tag for a in articles for tag in a['tags']}, key=str.casefold)
    archive_routes = (
        [(None, None, None)]
        + [(t, None, None) for t in topics]
        + [(None, tag, None) for tag in all_tags]
        + [(None, None, role) for role in CONTENT_ROLES]
    )
    for topic_id, active_tag, active_role in archive_routes:
        selected = [
            a for a in articles
            if (topic_id is None or a['topic_id'] == topic_id)
            and (active_tag is None or active_tag in a['tags'])
            and (active_role is None or a['role'] == active_role)
        ]
        label = TOPICS.get(topic_id, topic_id) if topic_id else '文章目录'
        route = f'topics/{topic_id}/' if topic_id else 'articles/'
        description = f'{label}相关文章：源码分析、执行链路与推理工程实践。' if topic_id else '按入门、进阶、深入研究和速查浏览，或按专题找到当前问题。'
        if active_tag:
            label = active_tag
            route = f'tags/{tag_slug(active_tag)}/'
            description = f'{label}相关文章：从基础原理到源码与性能实践。'
        if active_role:
            label = CONTENT_ROLES[active_role]['label']
            route = f'levels/{active_role}/'
            description = CONTENT_ROLES[active_role]['description']
        filters = f'<a href="{BASE}articles/"' + (' aria-current="page"' if not topic_id and not active_tag and not active_role else '') + '>全部文章</a>'
        filters += ''.join(f'<a href="{BASE}topics/{t}/"' + (' aria-current="page"' if t == topic_id else '') + f'>{esc(TOPICS.get(t,t))}</a>' for t in topics)
        role_filters = ''.join(
            f'<a href="{BASE}levels/{role}/"' + (' aria-current="page"' if role == active_role else '')
            + f'>{esc(spec["label"])}<span>{sum(a["role"] == role for a in articles)}</span></a>'
            for role, spec in CONTENT_ROLES.items()
        )
        tag_filters = ''.join(f'<a href="{BASE}tags/{tag_slug(t)}/"' + (' aria-current="page"' if active_tag == t else '') + f'>{esc(t)}<span>{sum(t in a["tags"] for a in articles)}</span></a>' for t in all_tags)
        if not topic_id and not active_tag and not active_role:
            selected.sort(key=lambda a: list(CONTENT_ROLES).index(a['role']))
        archive_cards = ''.join(card(a) for a in selected)
        page = f'''<main id="main" class="archive wrap"><div class="archive-heading"><div><p class="eyebrow">THE KNOWLEDGE SHELF</p><h1>{esc(label)}<span class="count">{len(selected):02d}</span></h1><p>{esc(description)}</p></div><a class="text-link" href="{BASE}series/">按顺序阅读 →</a></div><div class="archive-layout"><aside class="archive-sidebar" aria-label="文章分类"><details class="catalog-filters" open><summary>浏览分类<span aria-hidden="true">⌄</span></summary><div class="catalog-filter-body"><span class="filter-label">专题</span><nav class="topic-filters" aria-label="筛选专题">{filters}</nav><span class="filter-label">阅读层级</span><nav class="tag-filters role-filters" aria-label="筛选阅读层级">{role_filters}</nav><details class="tag-disclosure"{' open' if active_tag else ''}><summary>全部标签 <span>{len(all_tags)}</span></summary><nav class="tag-filters" aria-label="筛选内容标签">{tag_filters}</nav></details></div></details><a class="sidebar-map-link" href="{BASE}series/">阅读地图<span>按系列逐步深入 ↗</span></a></aside><section class="archive-results" aria-label="文章列表"><div class="catalog-tools" hidden><label class="catalog-query"><span class="sr-only">筛选当前目录</span><input id="catalog-query" type="search" placeholder="筛选标题、摘要、标签…" autocomplete="off"></label><label class="catalog-sort-label"><span class="sr-only">文章排序</span><select id="catalog-sort"><option value="recommended">推荐顺序</option><option value="updated">最近更新</option><option value="shortest">阅读时间短 → 长</option></select></label><button class="saved-filter" aria-pressed="false">稍后读</button><button class="catalog-reset" aria-label="重置筛选">重置</button></div><div class="catalog-status"><span data-catalog-count aria-live="polite">{len(selected)} 篇文章</span><button class="archive-search" hidden>搜索全文 ↗</button></div><div class="archive-list">{archive_cards}</div><div class="catalog-empty" hidden><span class="mascot mascot-lost" aria-hidden="true"></span><h2>没有找到符合条件的文章</h2><p>换个关键词，或清除“稍后读”筛选。收藏仅保存在当前浏览器。</p><button class="catalog-clear">清除筛选</button><a href="{BASE}articles/">查看全部文章</a></div></section></div></main>'''
        dest = OUT / route / 'index.html'
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(shell(label, page, 'archive-page', route, description, indexable=not bool(active_tag)), encoding='utf-8')
    (OUT / 'search.json').write_text(json.dumps(articles + skills, ensure_ascii=False), encoding='utf-8')
    (OUT / '.nojekyll').touch()
    credits = f'''<main id="main" class="credits wrap"><a class="back-link" href="{BASE}">← 首页</a><h1>插画来源</h1><p>本站为 AkiYang 的独立个人技术站，非 DeepSeek 官方网站。</p><h2>角色设计</h2><p>根据站点所有者提供的作者信息，鲸鱼娘角色设计署名 <a href="https://space.bilibili.com/4168597/dynamic">ZipZipPipe</a>。角色设计署名不等于下列每张衍生插画都由该作者绘制。</p><h2>首页看板</h2><p>首页宽幅鲸鱼娘看板以站点所有者提供的蓝发鲸尾女仆立绘作为角色与服装参考，经 AI 延展为蓝白科技海洋场景；该宽幅场景并非原画师原作，具体处理记录见素材来源文件。</p><h2>Q 版状态素材</h2><p>阅读、搜索、完成与迷路四种状态为 AI 生成的角色延展素材，非 ZipZipPipe 原作。不对原角色或用户提供的插画主张原创或再许可。</p><p class="credits-note">历史素材与处理记录见 <a href="{REPO}/blob/main/site/assets/SOURCES.md">SOURCES.md</a>。</p></main>'''
    (OUT / 'credits').mkdir()
    (OUT / 'credits/index.html').write_text(shell('插画来源', credits, 'credits-page', 'credits/'), encoding='utf-8')
    error = f'<main id="main" class="error-page wrap"><span class="mascot mascot-lost" aria-hidden="true"></span><span class="eyebrow">404 / PAGE NOT FOUND</span><h1>这一页，游到别处去了。</h1><p>链接可能已经变更，可以从文章目录继续阅读。</p><a class="button-primary" href="{BASE}articles/">浏览文章 →</a></main>'
    (OUT / '404.html').write_text(shell('页面未找到', error, route='404.html'), encoding='utf-8')
    feeds(OUT, articles, ORIGIN, BASE)
    print(f'Built {len(articles)} article(s), {len(topics)} topic(s), {len(skills)} skill(s) into {OUT}')

if __name__ == '__main__':
    build()
