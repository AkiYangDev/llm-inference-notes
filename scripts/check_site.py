"""Validate generated search metadata and local navigation before deployment."""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit, unquote
import json
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1] / '_site'
BASE = '/llm-inference-notes/'
ORIGIN = 'https://akiyangdev.github.io'
class Page(HTMLParser):
    def __init__(self, text):
        super().__init__(); self.meta = {}; self.links = []; self.schema = ''; self.in_schema = False; self.feed(text)
    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == 'meta': self.meta[a.get('name', a.get('property'))] = a.get('content', '')
        if tag in ('a', 'link') and 'href' in a: self.links.append(a)
        if tag == 'script' and a.get('type') == 'application/ld+json': self.in_schema = True
    def handle_endtag(self, tag):
        if tag == 'script': self.in_schema = False
    def handle_data(self, text):
        if self.in_schema: self.schema += text

urls = {node.text for node in ET.parse(ROOT / 'sitemap.xml').findall('.//{*}loc')}
pages = list(ROOT.rglob('*.html'))
for path in pages:
    relative = path.relative_to(ROOT).as_posix()
    route = relative.removesuffix('index.html')
    page = Page(path.read_text())
    canonical = [a['href'] for a in page.links if a.get('rel') == 'canonical']
    assert canonical == [ORIGIN + BASE + route], (relative, 'canonical')
    assert page.meta.get('description'), (relative, 'description')
    schema = json.loads(page.schema)
    assert schema['url'] == canonical[0]
    if relative == '404.html':
        assert 'noindex' in page.meta['robots'] and canonical[0] not in urls
    else:
        assert 'noindex' not in page.meta.get('robots', '') and canonical[0] in urls
    if relative.startswith('articles/') and relative != 'articles/index.html':
        assert schema['@type'] == 'BlogPosting'
    for a in page.links:
        url = urlsplit(a['href'])
        if not url.netloc and url.path.startswith(BASE):
            target = ROOT / unquote(url.path[len(BASE):])
            if url.path.endswith('/'): target /= 'index.html'
            assert target.exists(), (relative, a['href'])
assert len(urls) == len(pages) - 1
items = ET.parse(ROOT / 'feed.xml').findall('./channel/item')
assert len(items) == len(list((ROOT / 'articles').glob('*/*/index.html')))
print(f'Site checks passed: {len(pages)} pages, {len(urls)} sitemap URLs, {len(items)} feed entries.')
