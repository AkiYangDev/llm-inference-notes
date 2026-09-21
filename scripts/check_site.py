"""Validate generated search metadata and local navigation before deployment."""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit, unquote
import json
import xml.etree.ElementTree as ET

REPO_ROOT = Path(__file__).resolve().parents[1]
ROOT = REPO_ROOT / '_site'
BASE = '/llm-inference-notes/'
ORIGIN = 'https://akiyangdev.github.io'
_redirects_path = REPO_ROOT / 'site/redirects.json'
REDIRECTS = json.loads(_redirects_path.read_text()) if _redirects_path.exists() else {}
REDIRECT_INDEXES = {
    old.strip('/') + '/index.html': new.strip('/') + '/'
    for old, new in REDIRECTS.items()
}

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
    assert page.meta.get('description'), (relative, 'description')

    if relative in REDIRECT_INDEXES:
        target_route = REDIRECT_INDEXES[relative]
        target_url = ORIGIN + BASE + target_route
        assert canonical == [target_url], (relative, 'redirect canonical')
        assert 'noindex' in page.meta.get('robots', ''), (relative, 'redirect robots')
        assert target_url in urls, (relative, 'redirect target missing from sitemap')
    else:
        assert canonical == [ORIGIN + BASE + route], (relative, 'canonical')
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

assert len(urls) == len(pages) - 1 - len(REDIRECT_INDEXES)
items = ET.parse(ROOT / 'feed.xml').findall('./channel/item')
article_pages = [
    p for p in (ROOT / 'articles').glob('*/*/index.html')
    if p.relative_to(ROOT).as_posix() not in REDIRECT_INDEXES
]
assert len(items) == len(article_pages)
print(f'Site checks passed: {len(pages)} pages, {len(urls)} sitemap URLs, {len(items)} feed entries, {len(REDIRECT_INDEXES)} redirects.')

# Tag archives must contain exactly their members, and card links cannot nest.
import hashlib
articles = [a for a in json.loads((ROOT / 'search.json').read_text()) if a.get('kind') != 'skill']

# Every published Markdown document must have a generated article route.
source_docs = sorted(
    p for p in (REPO_ROOT / 'docs').rglob('*.md')
    if p.name != 'README.md'
)
expected_article_urls = {
    BASE + 'articles/' + p.parent.name + '/' + p.stem + '/'
    for p in source_docs
}
generated_article_urls = {a['url'] for a in articles}
assert generated_article_urls == expected_article_urls, (
    'article publication mismatch',
    sorted(expected_article_urls - generated_article_urls),
    sorted(generated_article_urls - expected_article_urls),
)

class Cards(HTMLParser):
    def __init__(self, text):
        super().__init__(); self.urls = []; self.active = []; self.depth = 0; self.in_tags = False; self.feed(text)
    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == 'nav' and a.get('class') == 'tag-filters': self.in_tags = True
        if tag == 'a':
            assert not self.depth, 'Nested anchors break tag navigation'
            self.depth += 1
            if 'card-main' in a.get('class', '').split(): self.urls.append(a['href'])
            if self.in_tags and a.get('aria-current') == 'page': self.active.append(a['href'])
    def handle_endtag(self, tag):
        if tag == 'a': self.depth -= 1
        if tag == 'nav': self.in_tags = False

for tag in {t for a in articles for t in a['tags']}:
    slug = hashlib.sha256(tag.encode('utf-8')).hexdigest()[:12]
    route = f'tags/{slug}/'
    parsed = Cards((ROOT / route / 'index.html').read_text())
    assert set(parsed.urls) == {a['url'] for a in articles if tag in a['tags']}, tag
    assert parsed.active == [BASE + route], (tag, parsed.active)

for page in pages: Cards(page.read_text())
print('Tag membership, selected filter and non-nested navigation checks passed.')
