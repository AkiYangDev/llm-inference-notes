"""Publication metadata derived from document history and explicit source evidence."""
import html
import json
import re
import subprocess
from datetime import datetime
from email.utils import format_datetime
from pathlib import Path
from xml.etree import ElementTree as ET

PAGES = {}

def history(root, path):
    try:
        lines = subprocess.check_output(['git', 'log', '--follow', '--format=%cI', '--', str(path.relative_to(root))], cwd=root, stderr=subprocess.DEVNULL, text=True).splitlines()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {}
    return {'published': lines[-1], 'modified': lines[0]} if lines else {}

def source_evidence(source):
    intro = re.split(r'^## ', source, maxsplit=1, flags=re.M)[0]
    return list(dict.fromkeys(re.findall(r'([\w.-]+/[\w.-]+)\s*@\s*([a-f0-9]{40})\b', intro)))

def evidence_panel(source, dates, repo, path):
    bits=[]
    for key,label in [('published','首次提交'),('modified','最近更新')]:
        if dates.get(key):
            date=dates[key];bits.append(f'<span>{label} <time datetime="{html.escape(date)}" title="{html.escape(date)}">{date[:10]}</time></span>')
    for name,sha in source_evidence(source):
        bits.append(f'<a href="https://github.com/{name}/tree/{sha}" target="_blank" rel="noopener noreferrer">源码依据：{html.escape(name)} @ {sha[:12]} ↗</a>')
    bits.append(f'<a href="{repo}/commits/main/{path}">文章修改记录 ↗</a>')
    return '<div class="publication-meta">'+''.join(bits)+'</div>'

def metadata(title, description, kind, route, origin, base, dates=None):
    dates=dates or {};url=origin+base+route
    article=kind=='article-page'
    data={'@context':'https://schema.org','@type':'BlogPosting' if article else ('WebSite' if not route else 'WebPage'),'name':title,'description':description,'url':url,'inLanguage':'zh-CN'}
    image=origin+base+'assets/social-cover.png'
    if article:
        data.update(headline=title,author={'@type':'Person','name':'AkiYang','url':'https://github.com/AkiYangDev'},mainEntityOfPage=url)
        for k,f in [('published','datePublished'),('modified','dateModified')]:
            if dates.get(k):data[f]=dates[k]
    data['image']=image
    fields={'og:title':title+' · AkiYang','og:description':description,'og:url':url,'og:type':'article' if article else 'website','og:locale':'zh_CN','og:site_name':'AkiYang · 推理工程手记'}
    # Shared cover with article-specific titles and descriptions.
    fields.update({'og:image':image,'og:image:alt':'AkiYang 推理工程手记，蓝色鲸尾与金色细线'})
    for k,f in [('published','article:published_time'),('modified','article:modified_time')]:
        if article and dates.get(k):fields[f]=dates[k]
    tags=''.join(f'<meta property="{k}" content="{html.escape(v,quote=True)}">' for k,v in fields.items())
    tags+=f'<meta name="twitter:card" content="{"summary_large_image"}"><meta name="twitter:title" content="{html.escape(title,quote=True)}"><meta name="twitter:description" content="{html.escape(description,quote=True)}">'
    tags+=f'<meta name="twitter:image" content="{image}">'
    tags+=f'<link rel="alternate" type="application/rss+xml" title="AkiYang · 技术文章" href="{origin}{base}feed.xml">'
    tags+='<script type="application/ld+json">'+json.dumps(data,ensure_ascii=False).replace('<','\\u003c')+'</script>'
    if route=='404.html':tags+='<meta name="robots" content="noindex">'
    else:
        tags+='<meta name="robots" content="index,follow,max-image-preview:large">'
        PAGES[url]=dates
    return tags

def feeds(out, articles, origin, base):
    ns='http://www.sitemaps.org/schemas/sitemap/0.9'
    ET.register_namespace('',ns);root=ET.Element('{'+ns+'}urlset')
    for url,dates in sorted(PAGES.items()):
        node=ET.SubElement(root,'{'+ns+'}url');ET.SubElement(node,'{'+ns+'}loc').text=url
        if dates.get('modified'):ET.SubElement(node,'{'+ns+'}lastmod').text=dates['modified']
    ET.ElementTree(root).write(out/'sitemap.xml',encoding='utf-8',xml_declaration=True)
    rss=ET.Element('rss',version='2.0');channel=ET.SubElement(rss,'channel')
    for tag,text in [('title','AkiYang · 推理工程手记'),('link',origin+base),('description','SGLang、Ascend 与 DeepSeek 推理工程文章'),('language','zh-CN')]:ET.SubElement(channel,tag).text=text
    for a in sorted(articles,key=lambda a:a.get('published',''),reverse=True):
        item=ET.SubElement(channel,'item');url=origin+a['url']
        for tag,text in [('title',a['title']),('link',url),('description',a['excerpt'])]:ET.SubElement(item,tag).text=text
        ET.SubElement(item,'guid',isPermaLink='true').text=url
        if a.get('published'):ET.SubElement(item,'pubDate').text=format_datetime(datetime.fromisoformat(a['published']))
    ET.ElementTree(rss).write(out/'feed.xml',encoding='utf-8',xml_declaration=True)


def verification_meta(root):
    """Only render a real Search Console verification token supplied by the owner."""
    config = root / 'site/search-console.json'
    token = json.loads(config.read_text()).get('google_site_verification', '') if config.exists() else ''
    if not isinstance(token, str) or (token and not re.fullmatch(r'[A-Za-z0-9_-]+', token)):
        raise ValueError('Invalid Google verification token')
    return f'<meta name="google-site-verification" content="{html.escape(token, quote=True)}">' if token else ''
