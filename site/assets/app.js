import {rankSearchResults} from './search.mjs';
function safeFragment(hash) { try { return decodeURIComponent(hash.slice(1)); } catch { return hash.slice(1); } }
const base = new URL('../', import.meta.url);
const $ = selector => document.querySelector(selector);
const theme = $('.theme-toggle');
const dark = () => document.documentElement.dataset.theme === 'dark';
const toast = message => { const node = $('.toast'); node.textContent = message; node.classList.add('show'); clearTimeout(toast.timer); toast.timer = setTimeout(() => node.classList.remove('show'), 2400); };
function themeLabel() { const label = dark() ? '切换浅色模式' : '切换深色模式'; theme.setAttribute('aria-label', label); theme.title = label; }
themeLabel();
let updateDiagrams = () => {};
theme.addEventListener('click', () => { const next = dark() ? 'light' : 'dark'; document.documentElement.dataset.theme = next; try { localStorage.setItem('aki-theme', next); } catch {} themeLabel(); updateDiagrams(); });
const media = matchMedia('(prefers-color-scheme: dark)');
media.addEventListener('change', event => { try { if (localStorage.getItem('aki-theme')) return; } catch {} document.documentElement.dataset.theme = event.matches ? 'dark' : 'light'; themeLabel(); updateDiagrams(); });

const search = $('#search-dialog'), input = $('#search-input'), results = $('#search-results');
let indexPromise, searchRevision = 0, searchTimer;
function highlight(node, value, query) {
  const terms = [...new Set(query.trim().toLowerCase().split(/\s+/).filter(Boolean))];
  if (!terms.length) { node.textContent = value; return; }
  const lower = value.toLowerCase();
  let cursor = 0;
  while (cursor < value.length) {
    const hits = terms.map(term => ({term, at: lower.indexOf(term, cursor)})).filter(hit => hit.at >= 0).sort((a,b) => a.at - b.at || b.term.length - a.term.length);
    if (!hits.length) { node.append(document.createTextNode(value.slice(cursor))); break; }
    const {term, at} = hits[0];
    node.append(document.createTextNode(value.slice(cursor, at)));
    const mark = document.createElement('mark'); mark.textContent = value.slice(at, at + term.length); node.append(mark);
    cursor = at + term.length;
  }
}
async function searchArticles() {
  const revision = ++searchRevision, query = input.value.trim(), needle = query.toLowerCase();
  try {
    if (!indexPromise) indexPromise = fetch(new URL('search.json', base), {cache:'no-cache'}).then(r => { if (!r.ok) throw Error(); return r.json(); }).catch(error => { indexPromise = null; throw error; });
    const articles = await indexPromise;
    if (revision !== searchRevision) return;
    const found = rankSearchResults(articles, query);
    results.replaceChildren();
    if (!found.length) {
      const empty = document.createElement('div'); empty.className = 'search-empty';
      const mascot = document.createElement('span'); mascot.className = 'mascot mascot-lost'; mascot.setAttribute('aria-hidden', 'true');
      const message = document.createElement('p'); message.textContent = '还没有找到相关内容，换一个关键词试试。'; empty.append(mascot, message); results.append(empty); return;
    }
    const count = document.createElement('p'); count.className = 'search-count'; count.textContent = `找到 ${found.length} 篇内容${found.length > 15 ? '，显示前 15 篇；可增加关键词缩小范围' : ''}`; results.append(count);
    for (const result of found.slice(0, 15)) {
      const link = document.createElement('a'); link.href = result.article.url + (result.anchor ? '#' + encodeURIComponent(result.anchor) : ''); link.className = 'search-result';
      const title = document.createElement('strong'); highlight(title, result.article.title, query); link.append(title); const type = document.createElement('span'); type.className = 'result-heading'; type.textContent = result.article.kind === 'skill' ? 'Skill · 介绍与使用' : '文章 · ' + result.article.topic; link.append(type);
      if (result.heading) { const heading = document.createElement('span'); heading.className = 'result-heading'; highlight(heading, result.heading + ' ↗', query); link.append(heading); }
      const excerpt = document.createElement('span'); excerpt.className = 'result-excerpt'; const positions = needle.split(/\s+/).filter(Boolean).map(term => result.text.toLowerCase().indexOf(term)).filter(at => at >= 0); const pos = positions.length ? Math.min(...positions) : 0; const start = Math.max(0, pos - 28); const snippet = (start ? '…' : '') + result.text.slice(start, start + 130) + (result.text.length > start + 130 ? '…' : ''); highlight(excerpt, snippet, query); link.append(excerpt);
      link.addEventListener('click', () => search.close()); results.append(link);
    }
  } catch { if (revision === searchRevision) results.textContent = '搜索暂时无法加载，请稍后重试，或从文章目录浏览。'; }
}
function openSearch() { if (search.open) return; search.showModal(); input.focus(); searchArticles(); }
$('.search-trigger').addEventListener('click', openSearch);
$('.archive-search')?.addEventListener('click', openSearch);
$('.close-search').addEventListener('click', () => search.close());
input.addEventListener('input', () => { clearTimeout(searchTimer); searchRevision++; searchTimer = setTimeout(searchArticles, 120); });
search.addEventListener('keydown', event => {
  const links = [...results.querySelectorAll('a')];
  if (event.key === 'ArrowDown' || event.key === 'ArrowUp') { event.preventDefault(); if (!links.length) return; const i = links.indexOf(document.activeElement); const next = event.key === 'ArrowDown' ? (i + 1) % links.length : (i <= 0 ? links.length - 1 : i - 1); links[next].focus(); }
  if (event.key === 'Enter' && document.activeElement === input && links.length) { event.preventDefault(); links[0].click(); }
});
document.addEventListener('keydown', event => { if (event.key === '/' && !/INPUT|TEXTAREA|SELECT/.test(document.activeElement.tagName) && !document.activeElement.isContentEditable && !document.querySelector('dialog[open]')) { event.preventDefault(); openSearch(); } });
for (const dialog of document.querySelectorAll('dialog')) dialog.addEventListener('click', event => { if (event.target === dialog) { const r = dialog.getBoundingClientRect(); if (event.clientX < r.left || event.clientX > r.right || event.clientY < r.top || event.clientY > r.bottom) dialog.close(); } });
async function copyText(text, button, success) { try { await navigator.clipboard.writeText(text); toast(success); if (button) { const previous = button.textContent; button.textContent = '已复制'; setTimeout(() => button.textContent = previous, 2000); } } catch { toast('复制未完成，请手动选择内容复制。'); } }
$('.copy-link')?.addEventListener('click', event => copyText(location.href, event.currentTarget, '页面链接已复制'));
for (const block of document.querySelectorAll('.prose pre:not(.mermaid)')) { const button = document.createElement('button'); button.className = 'copy-code'; button.textContent = '复制代码'; button.setAttribute('aria-label', '复制代码'); button.addEventListener('click', () => copyText(block.querySelector('code')?.textContent || '', button, '代码已复制')); block.append(button); }
for (const link of document.querySelectorAll('.mobile-toc a')) link.addEventListener('click', () => { const panel = link.closest('details'); panel.open = false; });
if ($('.prose')) {
  const progress = $('.reading-progress'), headings = [...document.querySelectorAll('.prose h2, .prose h3')], links = [...document.querySelectorAll('.toc a')];
  let scheduled = false;
  const update = () => { const doc = document.documentElement; progress.style.width = (doc.scrollTop / Math.max(1, doc.scrollHeight - doc.clientHeight) * 100) + '%'; let current = headings[0]; for (const heading of headings) { if (heading.getBoundingClientRect().top < $('.header').getBoundingClientRect().height + 110) current = heading; else break; } for (const link of links) { const selected = current && decodeURIComponent(link.hash.slice(1)) === current.id; link.classList.toggle('current', !!selected); if (selected) link.setAttribute('aria-current', 'location'); else link.removeAttribute('aria-current'); } scheduled = false; };
  window.addEventListener('scroll', () => { if (!scheduled) { scheduled = true; requestAnimationFrame(update); } }, {passive:true}); window.addEventListener('resize', update); update();
  // Highlighting and diagrams load independently; failure of one does not delay the other.
  (async () => { try { const {default:hljs} = await import('https://cdn.jsdelivr.net/npm/@highlightjs/cdn-assets@11.11.1/es/highlight.min.js'); const css = document.createElement('link'); css.rel = 'stylesheet'; css.href = 'https://cdn.jsdelivr.net/npm/@highlightjs/cdn-assets@11.11.1/styles/github-dark.min.css'; document.head.append(css); document.querySelectorAll('.prose pre code').forEach(c => hljs.highlightElement(c)); } catch {} })();
  (async () => {
    const diagrams = [...document.querySelectorAll('.mermaid')].map(node => ({node, source:node.textContent}));
    if (!diagrams.length) return;
    try {
      const {default:mermaid} = await import('https://cdn.jsdelivr.net/npm/mermaid@11.4.1/dist/mermaid.esm.min.mjs');
      let rendering = false, pending = false, generation = 0;
      updateDiagrams = async () => {
        if (rendering) { pending = true; return; } rendering = true;
        do {
          pending = false; const isDark = dark();
          mermaid.initialize({startOnLoad:false,securityLevel:'strict',theme:'base',themeVariables:{primaryColor:isDark?'#263a59':'#eaf0ff',primaryTextColor:isDark?'#e6edf9':'#243554',primaryBorderColor:isDark?'#6e8fbb':'#9aafd9',lineColor:isDark?'#a4b6d0':'#6a7f9e',secondaryColor:isDark?'#1b2a42':'#f4f7fc',tertiaryColor:isDark?'#22314b':'#ffffff',edgeLabelBackground:isDark?'#19263a':'#f4f7fc',clusterBkg:isDark?'#17263c':'#f4f7fc',clusterBorder:isDark?'#405777':'#d0daec',fontFamily:'sans-serif',fontSize:'15px'},flowchart:{htmlLabels:false}});
          for (const {node, source} of diagrams) {
            try { const result = await mermaid.render('engineering-graph-' + (++generation), source); node.innerHTML = result.svg; const svg = node.querySelector('svg'); svg?.setAttribute('role','img'); svg?.setAttribute('aria-label','本节工程流程图'); }
            catch { node.textContent = source; if (!node.parentElement.querySelector('.diagram-error')) { const p = document.createElement('p'); p.className = 'diagram-error'; p.textContent = '图表暂时无法绘制，以下保留原始描述。'; node.before(p); } }
          }
        } while (pending);
        rendering = false;
        // Re-align deep links after diagrams change the document height.
        if (location.hash && !updateDiagrams.aligned) { updateDiagrams.aligned = true; requestAnimationFrame(() => requestAnimationFrame(() => { const target = document.getElementById(safeFragment(location.hash)); if (target) window.scrollTo({top:window.scrollY + target.getBoundingClientRect().top - $('.header').getBoundingClientRect().height - 24, behavior:'instant'}); })); }
      };
      await updateDiagrams();
      for (const {node} of diagrams) { const button = document.createElement('button'); button.textContent = '放大查看'; button.className = 'expand-diagram'; button.addEventListener('click', () => { const svg = node.querySelector('svg'); if (!svg) return; $('#diagram-view').replaceChildren(svg.cloneNode(true)); $('#diagram-dialog').showModal(); zoom = 1; applyZoom(); }); node.parentElement.append(button); }
    } catch { for (const {node} of diagrams) { const p = document.createElement('p'); p.className = 'diagram-error'; p.textContent = '图表暂时无法加载，以下保留 Mermaid 源码。'; node.before(p); } }
  })();
}
let zoom = 1;
function applyZoom() { const view = $('#diagram-view'), svg = view.querySelector('svg'); if (svg) { const box = svg.viewBox.baseVal; const room = Math.max(150, innerHeight * .68 - 40); const ratio = box.width && box.height ? box.width / box.height : 1; const fitWidth = Math.min(Math.max(100, view.clientWidth - 40), room * ratio); svg.style.width = fitWidth * zoom + 'px'; } $('.zoom-out').disabled = zoom <= .5; $('.zoom-in').disabled = zoom >= 3; }
$('.zoom-in').addEventListener('click', () => { zoom = Math.min(3, zoom + .25); applyZoom(); });
$('.zoom-out').addEventListener('click', () => { zoom = Math.max(.5, zoom - .25); applyZoom(); });
$('.zoom-reset').addEventListener('click', () => { zoom = 1; applyZoom(); });
$('.close-diagram').addEventListener('click', () => $('#diagram-dialog').close());
window.addEventListener('resize', () => { if ($('#diagram-dialog').open) applyZoom(); });

// Device-local reading state; completion is an explicit reader action.
const readingKey = 'aki-reading-v1';
let readingState = {};
try { const value = JSON.parse(localStorage.getItem(readingKey) || '{}'); if (value && typeof value === 'object' && !Array.isArray(value)) readingState = value; } catch {}
function refreshReading() {
  document.querySelectorAll('.article-card[data-url]').forEach(card => { const badge = card.querySelector('.read-badge'); if (badge) badge.hidden = readingState[card.dataset.url] !== true; });
  document.querySelectorAll('[data-reading-group]').forEach(group => {
    const items = [...group.querySelectorAll('[data-reading-item]')];
    let count = 0;
    items.forEach(item => {
      const done = readingState[item.dataset.readingItem] === true;
      item.classList.toggle('is-read', done);
      item.querySelector('[data-reading-status]').textContent = done ? '已读 ✓' : '未读';
      if (done) count++;
    });
    group.querySelector('[data-reading-summary]').textContent = `已读 ${count} / ${items.length} 篇`;
    const meter = group.querySelector('.map-progress');
    if (meter) { meter.setAttribute('aria-valuenow', String(count)); meter.querySelector('.map-progress-fill').style.transform = `scaleX(${items.length ? count/items.length : 0})`; }
    const next = items.find(item => readingState[item.dataset.readingItem] !== true);
    const link = group.querySelector('[data-continue-reading]');
    link.href = (next || items[0]).dataset.readingItem;
    link.textContent = count === items.length ? '重新阅读 →' : count ? '继续阅读 →' : '开始阅读 →';
  });
  document.querySelectorAll('.reading-complete').forEach(button => {
    const done = readingState[button.dataset.articleUrl] === true;
    button.setAttribute('aria-pressed', String(done));
    button.textContent = done ? '已读 ✓ · 撤销标记' : '标记为已读';
  });
}
document.querySelectorAll('.reading-complete').forEach(button => button.addEventListener('click', () => {
  const path = button.dataset.articleUrl;
  const done = readingState[path] !== true;
  const next = {...readingState, [path]: done};
  try { localStorage.setItem(readingKey, JSON.stringify(next)); readingState = next; refreshReading(); toast(done ? '已记录阅读进度' : '已取消已读标记'); }
  catch { toast('浏览器无法保存进度，请检查存储设置'); }
}));
window.addEventListener('storage', event => {
  if (event.key !== readingKey && event.key !== null) return;
  try { const value = JSON.parse(event.newValue || '{}'); readingState = value && typeof value === 'object' && !Array.isArray(value) ? value : {}; } catch { readingState = {}; }
  refreshReading();
});
window.addEventListener('pageshow', () => {
  try { const value = JSON.parse(localStorage.getItem(readingKey) || '{}'); readingState = value && typeof value === 'object' && !Array.isArray(value) ? value : {}; } catch { readingState = {}; }
  refreshReading();
});
refreshReading();

// A line address highlights code without altering its copied text or syntax markup.
function codeLines() {
  document.querySelectorAll('.prose pre > code').forEach((code, index) => {
    const pre = code.parentElement;
    const id = `code-${index + 1}`;
    pre.id = id;
    pre.classList.add('numbered-code');
    const gutter = document.createElement('div'); gutter.className = 'code-gutter'; gutter.setAttribute('aria-label', '代码行号，点击高亮并定位');
    const count = code.textContent.replace(/\n$/, '').split('\n').length;
    for (let line = 1; line <= count; line++) {
      const a = document.createElement('a'); a.href = `#${id}-L${line}`; a.id = `${id}-L${line}`; a.textContent = String(line); a.setAttribute('aria-label', `代码块 ${index + 1} 第 ${line} 行`);
      a.addEventListener('click', () => toast('已定位此行，可复制页面链接分享'));
      gutter.append(a);
    }
    pre.append(gutter);
  });
  highlightCodeLine();
}
function highlightCodeLine() {
  document.querySelectorAll('.numbered-code').forEach(pre => { pre.style.removeProperty('--selected-line'); pre.querySelectorAll('.code-gutter a').forEach(a => a.removeAttribute('aria-current')); });
  const match = location.hash.match(/^#(code-\d+)-L(\d+)$/);
  if (!match) return;
  const pre = document.getElementById(match[1]);
  const line = document.getElementById(`${match[1]}-L${match[2]}`);
  if (pre && line) { pre.style.setProperty('--selected-line', Number(match[2]) - 1); line.setAttribute('aria-current', 'location'); line.scrollIntoView({block:'center'}); }
}
codeLines();
window.addEventListener('hashchange', highlightCodeLine);

// Deliberate shortcuts: no navigation while typing, selecting, or using dialogs.
document.addEventListener('keydown', event => {
  if (!event.altKey || event.ctrlKey || event.metaKey || !event.shiftKey || event.repeat || !['ArrowLeft','ArrowRight'].includes(event.key)) return;
  const active = document.activeElement;
  if (/INPUT|TEXTAREA|SELECT/.test(active.tagName) || active.isContentEditable || document.querySelector('dialog[open]') || String(window.getSelection())) return;
  const link = document.querySelector(`.series-pager [rel="${event.key === 'ArrowLeft' ? 'prev' : 'next'}"]`);
  if (link) { event.preventDefault(); location.assign(link.href); }
});
if ($('.series-pager')) { const hint = document.createElement('p'); hint.className = 'reading-shortcuts'; hint.textContent = '系列切换：Alt + Shift + ← 上一篇 · Alt + Shift + → 下一篇'; $('.series-pager').after(hint); }

// Remember a stable section anchor and its local offset, not the whole page height.
if (document.body.classList.contains('article-page') && !document.body.classList.contains('skill-page')) {
  const positionKey = 'aki-position-v1:' + location.pathname;
  const panel = $('.resume-reading');
  let previous;
  try { previous = JSON.parse(localStorage.getItem(positionKey) || 'null'); } catch {}
  if (previous && typeof previous.anchor === 'string' && document.getElementById(previous.anchor) && Number.isFinite(previous.offset) && !location.hash) {
    panel.hidden = false;
    $('.resume-position').addEventListener('click', () => {
      const heading = document.getElementById(previous.anchor);
      window.scrollTo({top:Math.max(0, window.scrollY + heading.getBoundingClientRect().top + Math.min(Math.max(previous.offset,0),3000)),behavior:'instant'});
      panel.hidden = true;
    });
  }
  let timer, allowSave=true;
  $('.forget-position')?.addEventListener('click', () => { try {localStorage.removeItem(positionKey);} catch {} panel.hidden=true; allowSave=false; clearTimeout(timer); toast('已清除上次阅读位置'); });
  function savePosition() {
    if (!allowSave || window.scrollY < 250) return;
    const headings=[...document.querySelectorAll('.prose h2,.prose h3')];
    const heading=headings.filter(h=>h.getBoundingClientRect().top<=100).at(-1);
    if (heading) try {localStorage.setItem('aki-last-reading-v1', JSON.stringify({url:location.pathname,title:$('.article-header h1').textContent})); localStorage.setItem(positionKey,JSON.stringify({anchor:heading.id,offset:Math.max(0,-heading.getBoundingClientRect().top)}));}catch {}
  }
  window.addEventListener('scroll',()=>{clearTimeout(timer);timer=setTimeout(savePosition,450);},{passive:true});
  window.addEventListener('pagehide',savePosition);
}

// Load the formula renderer only on pages that contain display mathematics.
if (document.querySelector('.math-block')) {
  (async () => {
    try {
      const {default: katex} = await import('https://cdn.jsdelivr.net/npm/katex@0.16.22/dist/katex.mjs');
      const css = document.createElement('link'); css.rel = 'stylesheet'; css.href = 'https://cdn.jsdelivr.net/npm/katex@0.16.22/dist/katex.min.css'; document.head.append(css);
      document.querySelectorAll('.math-block').forEach(node => {
        katex.render(node.textContent, node, {displayMode:true, throwOnError:false, trust:false, output:'htmlAndMathml'});
      });
    } catch { /* Preserve readable TeX if the CDN is unavailable. */ }
  })();
}

// Progressive enhancements remain optional: all article links work without storage or JS.
const menu = $('#menu-dialog'), menuButton = $('.menu-toggle');
menuButton.hidden = false;
menuButton.addEventListener('click', () => menu.showModal());
$('.close-menu').addEventListener('click', () => menu.close());
for (const button of document.querySelectorAll('[data-search-query]')) button.addEventListener('click', () => { input.value = button.dataset.searchQuery; input.focus(); searchArticles(); });
document.addEventListener('keydown', event => {
  if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'k' && !event.altKey) {
    if (document.querySelector('dialog[open]') && !search.open) return;
    event.preventDefault(); if (search.open) search.close(); else openSearch();
  }
});
function readObject(key) { try { const value = JSON.parse(localStorage.getItem(key)); return value && typeof value === 'object' && !Array.isArray(value) ? value : {}; } catch { return {}; } }
const bookmarkKey = 'aki-bookmarks-v1';
let bookmarks = readObject(bookmarkKey), refreshCatalog = () => {};
function refreshBookmarks() {
  document.querySelectorAll('[data-bookmark]').forEach(button => {
    const saved = bookmarks[button.dataset.bookmark] === true;
    button.hidden = false; button.setAttribute('aria-pressed', String(saved));
    button.setAttribute('aria-label', (saved ? '取消收藏：' : '稍后阅读：') + button.dataset.title);
    button.title = saved ? '取消收藏' : '稍后阅读'; button.querySelector('span').textContent = saved ? '已收藏' : '稍后读';
  });
  refreshCatalog();
}
for (const button of document.querySelectorAll('[data-bookmark]')) button.addEventListener('click', () => {
  const next = {...readObject(bookmarkKey)}, url = button.dataset.bookmark;
  if (next[url] === true) delete next[url]; else next[url] = true;
  try { localStorage.setItem(bookmarkKey, JSON.stringify(next)); bookmarks = next; refreshBookmarks(); toast(next[url] ? '已加入稍后读 · 仅保存在当前浏览器' : '已取消收藏'); } catch { toast('浏览器无法保存收藏，请检查存储设置'); }
});
function refreshLastReading() {
  const link = $('[data-last-reading]'); if (!link) return;
  const record = readObject('aki-last-reading-v1'); link.hidden = true;
  try { const url = new URL(record.url, location.origin); if (typeof record.title !== 'string' || !record.title || url.origin !== location.origin || !url.pathname.startsWith(base.pathname + 'articles/') || url.pathname === base.pathname + 'articles/') return;
    link.href = url.pathname; link.querySelector('strong').textContent = record.title.slice(0, 200); link.hidden = false;
  } catch {}
}
window.addEventListener('storage', event => { if (event.key === bookmarkKey || event.key === null) { bookmarks = readObject(bookmarkKey); refreshBookmarks(); } refreshLastReading(); });
window.addEventListener('pageshow', () => { bookmarks = readObject(bookmarkKey); refreshBookmarks(); refreshLastReading(); });
refreshBookmarks(); refreshLastReading();

const catalog = $('.archive-list');
if (catalog) {
  const cards = [...catalog.querySelectorAll('.article-card')], query = $('#catalog-query'), sort = $('#catalog-sort'), savedButton = $('.saved-filter');
  let savedOnly = false;
  $('.catalog-tools').hidden = false; $('.archive-search').hidden = false;
  function restoreFilters() { const params = new URL(location.href).searchParams; query.value = params.get('q') || ''; sort.value = ['updated','shortest'].includes(params.get('sort')) ? params.get('sort') : 'recommended'; savedOnly = params.get('saved') === '1'; refreshCatalog(); }
  refreshCatalog = () => {
    const terms = query.value.toLocaleLowerCase().trim().split(/\s+/).filter(Boolean);
    const ordered = [...cards];
    if (sort.value === 'updated') ordered.sort((a,b) => b.dataset.modified.localeCompare(a.dataset.modified));
    if (sort.value === 'shortest') ordered.sort((a,b) => Number(a.dataset.minutes) - Number(b.dataset.minutes));
    let count = 0;
    for (const card of ordered) { card.hidden = !terms.every(term => card.dataset.search.toLocaleLowerCase().includes(term)) || (savedOnly && bookmarks[card.dataset.url] !== true); if (!card.hidden) count++; catalog.append(card); }
    $('[data-catalog-count]').textContent = count + ' / ' + cards.length + ' 篇文章'; $('.catalog-empty').hidden = count !== 0; savedButton.setAttribute('aria-pressed', String(savedOnly));
  };
  function updateFilters() { refreshCatalog(); const url = new URL(location.href); for (const [key,value] of [['q',query.value.trim()],['sort',sort.value === 'recommended' ? '' : sort.value],['saved',savedOnly ? '1' : '']]) { if (value) url.searchParams.set(key,value); else url.searchParams.delete(key); } history.replaceState(null,'',url); }
  query.addEventListener('input', updateFilters); sort.addEventListener('change', updateFilters);
  savedButton.addEventListener('click', () => { savedOnly = !savedOnly; updateFilters(); });
  const reset = () => { query.value = ''; sort.value = 'recommended'; savedOnly = false; updateFilters(); query.focus(); };
  $('.catalog-reset').addEventListener('click', reset); $('.catalog-clear').addEventListener('click', reset);
  window.addEventListener('popstate', restoreFilters); restoreFilters();
  if (matchMedia('(max-width:800px)').matches) $('.catalog-filters').open = false;
}
const toolbar = $('.reading-toolbar');
if (toolbar) {
  toolbar.hidden = false;
  let fontSize = 18;
  try { const value = Number(localStorage.getItem('aki-font-size-v1')); if (Number.isInteger(value) && value >= 16 && value <= 22) fontSize = value; } catch {}
  function updateFont() { document.documentElement.style.setProperty('--prose-size', fontSize + 'px'); $('.font-value').textContent = String(fontSize); $('.font-smaller').disabled = fontSize <= 16; $('.font-larger').disabled = fontSize >= 22; }
  for (const [selector,delta] of [['.font-smaller',-1],['.font-larger',1]]) $(selector).addEventListener('click', () => { fontSize = Math.min(22,Math.max(16,fontSize+delta)); updateFont(); try { localStorage.setItem('aki-font-size-v1',String(fontSize)); } catch {} });
  updateFont();
  $('.focus-toggle').addEventListener('click', event => { const focused = document.body.classList.toggle('focus-reading'); event.currentTarget.setAttribute('aria-pressed',String(focused)); event.currentTarget.textContent = focused ? '退出专注' : '专注阅读'; });
}
for (const heading of document.querySelectorAll('.prose h2[id],.prose h3[id]')) {
  const anchor = document.createElement('a'); anchor.className = 'heading-anchor'; anchor.href = '#' + encodeURIComponent(heading.id); anchor.textContent = '#'; anchor.setAttribute('aria-label','复制本节链接：' + heading.textContent);
  anchor.addEventListener('click', event => { event.preventDefault(); copyText(anchor.href,null,'本节链接已复制'); }); heading.append(anchor);
}
