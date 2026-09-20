"""Build a static reading site from the repository's published Markdown."""
from pathlib import Path
import html
import hashlib
import json
import re
import shutil
import markdown

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / '_site'
BASE = '/llm-inference-notes/'
REPO = 'https://github.com/AkiYangDev/llm-inference-notes'
TOPICS = {'sglang': 'SGLang', 'ascend': 'Ascend', 'fundamentals': '推理基础', 'distributed': '分布式推理', 'speculative-decoding': '投机推理', 'performance': '性能分析'}

def esc(value):
    return html.escape(str(value), quote=True)

def shell(title, body, kind='home'):
    asset_version = hashlib.sha256((ROOT / 'site/assets/style.css').read_bytes() + (ROOT / 'site/assets/app.js').read_bytes()).hexdigest()[:12]
    return f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(title)} · AkiYang</title><meta name="description" content="AkiYang 的大模型推理工程文档：SGLang、Ascend、源码与性能分析。">
<meta name="color-scheme" content="light dark"><link rel="icon" href="{BASE}assets/favicon.svg" type="image/svg+xml">
<link rel="stylesheet" href="{BASE}assets/style.css?v={asset_version}"><script>try{{if(localStorage.getItem('aki-theme')==='dark')document.documentElement.dataset.theme='dark'}}catch(e){{}}</script>
<script type="module" src="{BASE}assets/app.js?v={asset_version}"></script></head><body class="{kind}">
<a class="skip" href="#main">跳至正文</a><header class="header"><div class="header-inner">
<a class="brand" href="{BASE}" aria-label="AkiYang 首页"><span class="brand-icon">A<span>.</span></span><span>AkiYang<span class="brand-caption">ENGINEERING NOTES</span></span></a>
<nav aria-label="主导航"><a href="{BASE}#articles">文章</a><a href="{REPO}" target="_blank" rel="noopener noreferrer">GitHub <span aria-hidden="true">↗</span></a><button class="search-trigger" type="button" aria-label="搜索文章">搜索 <kbd>/</kbd></button><button class="theme-toggle" aria-label="切换深色模式" title="切换深色模式">◐</button></nav></div></header>
{body}
<footer class="footer"><a href="{BASE}">AkiYang<span> / INFERENCE ARCHIVE</span></a><span>独立个人技术站 · 非 DeepSeek 官方网站</span><a href="{REPO}">源代码 ↗</a></footer>
<dialog id="search-dialog"><div class="search-top"><label for="search-input">搜索文章</label><button class="close-search" aria-label="关闭搜索">×</button></div><input id="search-input" type="search" placeholder="搜索标题或正文，例如 KV Cache" autocomplete="off"><div id="search-results" aria-live="polite"></div><p class="dialog-hint">Esc 关闭 · 搜索覆盖已发布文章的正文</p></dialog>
<dialog id="diagram-dialog"><div class="search-top"><span>工程图</span><button class="close-diagram" aria-label="关闭工程图">×</button></div><div id="diagram-view"></div></dialog>
</body></html>'''

def build():
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / 'assets').mkdir(parents=True)
    for path in (ROOT / 'site' / 'assets').iterdir():
        shutil.copy2(path, OUT / 'assets' / path.name)
    articles = []
    for path in sorted((ROOT / 'docs').rglob('*.md')):
        if path.name == 'README.md':
            continue
        source = path.read_text(encoding='utf-8')
        title = re.search(r'^# (.+)$', source, re.M).group(1)
        topic = TOPICS.get(path.parent.name, path.parent.name)
        # Include the topic in the URL so equal filenames in different topics do not collide.
        slug = path.parent.name + '/' + path.stem
        url = BASE + 'articles/' + slug + '/'
        minutes = max(1, round(len(re.sub(r'```[\s\S]*?```', '', source)) / 650))
        md = markdown.Markdown(extensions=['fenced_code', 'tables', 'toc', 'sane_lists'], extension_configs={'toc': {'permalink': False}})
        rendered = md.convert(re.sub(r'^# .+\n', '', source, count=1))
        rendered = re.sub(r'<pre><code class="language-mermaid">([\s\S]*?)</code></pre>', r'<div class="diagram"><pre class="mermaid">\1</pre></div>', rendered)
        rendered = re.sub(r'(<table>[\s\S]*?</table>)', r'<div class="table-scroll">\1</div>', rendered)
        # Preserve repository-relative links on the generated article routes.
        def link(match):
            attr, destination = match.group(1), html.unescape(match.group(2))
            if destination.startswith(('http:', 'https:', '#', 'mailto:', 'data:', '/')):
                return match.group(0)
            from urllib.parse import quote
            dest = (path.parent / destination.split('#')[0]).resolve().relative_to(ROOT).as_posix()
            base = REPO + '/blob/main/' if attr == 'href' else 'https://raw.githubusercontent.com/AkiYangDev/llm-inference-notes/main/'
            return attr + '="' + base + quote(dest) + ('#' + destination.split('#', 1)[1] if '#' in destination else '') + '"'
        rendered = re.sub(r'(href|src)="([^"]+)"', link, rendered)
        body = f'''<div class="reading-progress" aria-hidden="true"></div><main id="main" class="article-layout"><aside class="article-nav"><a href="{BASE}#articles">← 全部文章</a><span class="eyebrow">IN THIS ARCHIVE</span><a class="active-topic" href="{BASE}#articles">{esc(topic)}</a><p>源码 · 执行链路</p></aside>
<div class="article-column"><header class="article-header"><div class="eyebrow">{esc(topic)} / ENGINEERING</div><h1>{esc(title)}</h1><div class="article-meta"><span>Aki Yang</span><span>约 {minutes} 分钟阅读</span><a href="{REPO}/blob/main/{path.relative_to(ROOT).as_posix()}">查看 Markdown ↗</a></div></header><details class="mobile-toc"><summary>本页目录</summary>{md.toc}</details><article class="prose">{rendered}</article><div class="article-end"><span>END OF ARTICLE</span><a href="{BASE}#articles">返回文章目录 ↗</a></div></div><aside class="toc-panel"><span class="eyebrow">本页目录</span>{md.toc}</aside></main>'''
        output = OUT / 'articles' / slug / 'index.html'
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(shell(title, body, 'article-page'), encoding='utf-8')
        plain = re.sub(r'<[^>]+>', ' ', rendered)
        articles.append({'title': title, 'topic': topic, 'url': url, 'minutes': minutes, 'text': html.unescape(plain)})
    cards = ''.join(f'''<a class="article-card" href="{a['url']}"><div class="card-meta"><span>{esc(a['topic'])}</span><span>源码解析 · 约 {a['minutes']} 分钟</span></div><h3>{esc(a['title'])}</h3><p>沿一次真实请求，理解调度、缓存映射、模型执行与设备算子的衔接。</p><div class="card-bottom"><span>阅读全文</span><span class="circle-arrow" aria-hidden="true">↗</span></div></a>''' for a in articles)
    featured_url = articles[0]['url'] if articles else BASE + '#articles'
    body = f'''<main id="main"><section class="hero wrap"><div class="hero-copy"><div class="eyebrow"><span class="tiny-line"></span> AKIYANG / TECHNICAL JOURNAL</div><h1>理解系统。<br>深入<span>每一次推理。</span></h1><p class="hero-description">从请求到算子，从源码到工程。<br>关于大模型推理的原理、实现与实践。</p><div class="hero-actions"><a class="button-primary" href="{featured_url}">阅读专题 <span aria-hidden="true">↗</span></a><a class="text-link" href="{REPO}">浏览 GitHub <span aria-hidden="true">↗</span></a></div><div class="hero-topics"><span>SGLang</span><span>Ascend NPU</span><span>LLM Inference</span></div></div><div class="hero-art"><span class="art-caption">DEEP BLUE / 01</span><img src="{BASE}assets/whale.jpg" alt="蓝白配色的 DeepSeek 鲸鱼娘主题插画" width="690" height="1215" fetchpriority="high"><span class="art-bottom">CODE MEETS CURIOSITY</span></div></section>
<section id="articles" class="articles-section wrap"><div class="section-heading"><div><span class="eyebrow">THE ARCHIVE</span><h2>技术文章<span class="count">{len(articles):02d}</span></h2></div><p>把复杂的执行过程，讲清楚。</p></div><div class="article-grid">{cards}<div class="archive-note"><span class="note-number">01—06</span><h3>从原理到执行</h3><p>推理基础 / SGLang 源码 / Ascend 部署<br>分布式推理 / 投机推理 / 性能分析</p><a href="{REPO}/tree/main/docs">浏览专题目录 ↗</a></div></div></section></main>'''
    (OUT / 'index.html').write_text(shell('推理工程手记', body), encoding='utf-8')
    (OUT / 'search.json').write_text(json.dumps(articles, ensure_ascii=False), encoding='utf-8')
    (OUT / '.nojekyll').touch()
    (OUT / '404.html').write_text(shell('页面未找到', '<main id="main" class="wrap error-page"><span class="eyebrow">404 / PAGE NOT FOUND</span><h1>这页档案不在这里。</h1><p>链接可能已变更，你可以从文章目录重新查找。</p><a class="button-primary" href="'+BASE+'">返回首页 ↗</a></main>'), encoding='utf-8')
    print(f'Built {len(articles)} article(s) into {OUT}')

if __name__ == '__main__':
    build()
