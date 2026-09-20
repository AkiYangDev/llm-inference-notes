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
  if (!query) { node.textContent = value; return; }
  let cursor = 0, at;
  const lower = value.toLowerCase(), needle = query.toLowerCase();
  while ((at = lower.indexOf(needle, cursor)) !== -1) { node.append(document.createTextNode(value.slice(cursor, at))); const mark = document.createElement('mark'); mark.textContent = value.slice(at, at + query.length); node.append(mark); cursor = at + query.length; }
  node.append(document.createTextNode(value.slice(cursor)));
}
async function searchArticles() {
  const revision = ++searchRevision, query = input.value.trim(), needle = query.toLowerCase();
  try {
    if (!indexPromise) indexPromise = fetch(new URL('search.json', base), {cache:'no-cache'}).then(r => { if (!r.ok) throw Error(); return r.json(); }).catch(error => { indexPromise = null; throw error; });
    const articles = await indexPromise;
    if (revision !== searchRevision) return;
    const found = [];
    for (const article of articles) {
      if (!needle || article.title.toLowerCase().includes(needle) || article.topic.toLowerCase().includes(needle) || (article.tags || []).some(tag => tag.toLowerCase().includes(needle))) found.push({article, heading:'', text:article.excerpt, anchor:'', score:3});
      if (needle) for (const section of article.sections) { const headingMatch = section.heading.toLowerCase().includes(needle); if (headingMatch || section.text.toLowerCase().includes(needle)) found.push({article, ...section, score:headingMatch ? 2 : 1}); }
    }
    found.sort((a,b) => b.score - a.score);
    results.replaceChildren();
    if (!found.length) {
      const empty = document.createElement('div'); empty.className = 'search-empty';
      const mascot = document.createElement('span'); mascot.className = 'mascot mascot-lost'; mascot.setAttribute('aria-hidden', 'true');
      const message = document.createElement('p'); message.textContent = '还没有找到相关内容，换一个关键词试试。'; empty.append(mascot, message); results.append(empty); return;
    }
    for (const result of found.slice(0, 15)) {
      const link = document.createElement('a'); link.href = result.article.url + (result.anchor ? '#' + encodeURIComponent(result.anchor) : ''); link.className = 'search-result';
      const title = document.createElement('strong'); highlight(title, result.article.title, query); link.append(title); const type = document.createElement('span'); type.className = 'result-heading'; type.textContent = result.article.kind === 'skill' ? 'Skill · 介绍与使用' : '文章 · ' + result.article.topic; link.append(type);
      if (result.heading) { const heading = document.createElement('span'); heading.className = 'result-heading'; highlight(heading, result.heading + ' ↗', query); link.append(heading); }
      const excerpt = document.createElement('span'); excerpt.className = 'result-excerpt'; const pos = result.text.toLowerCase().indexOf(needle); const start = Math.max(0, pos - 28); const snippet = (start ? '…' : '') + result.text.slice(start, start + 130) + (result.text.length > start + 130 ? '…' : ''); highlight(excerpt, snippet, query); link.append(excerpt);
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
  const update = () => { const doc = document.documentElement; progress.style.width = (doc.scrollTop / Math.max(1, doc.scrollHeight - doc.clientHeight) * 100) + '%'; let current = headings[0]; for (const heading of headings) { if (heading.getBoundingClientRect().top < $('.header').getBoundingClientRect().height + 55) current = heading; else break; } for (const link of links) { const selected = current && decodeURIComponent(link.hash.slice(1)) === current.id; link.classList.toggle('current', !!selected); if (selected) link.setAttribute('aria-current', 'location'); else link.removeAttribute('aria-current'); } scheduled = false; };
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
        if (location.hash && !updateDiagrams.aligned) { updateDiagrams.aligned = true; requestAnimationFrame(() => requestAnimationFrame(() => { const target = document.getElementById(decodeURIComponent(location.hash.slice(1))); if (target) window.scrollTo({top:window.scrollY + target.getBoundingClientRect().top - $('.header').getBoundingClientRect().height - 24, behavior:'instant'}); })); }
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
  if (!event.altKey || event.ctrlKey || event.metaKey || event.shiftKey || event.repeat || !['ArrowLeft','ArrowRight'].includes(event.key)) return;
  const active = document.activeElement;
  if (/INPUT|TEXTAREA|SELECT/.test(active.tagName) || active.isContentEditable || document.querySelector('dialog[open]') || String(window.getSelection())) return;
  const link = document.querySelector(`.series-pager [rel="${event.key === 'ArrowLeft' ? 'prev' : 'next'}"]`);
  if (link) { event.preventDefault(); location.assign(link.href); }
});
if ($('.series-pager')) { const hint = document.createElement('p'); hint.className = 'reading-shortcuts'; hint.textContent = '系列切换：Alt + ← 上一篇 · Alt + → 下一篇'; $('.series-pager').after(hint); }

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
    if (heading) try {localStorage.setItem(positionKey,JSON.stringify({anchor:heading.id,offset:Math.max(0,-heading.getBoundingClientRect().top)}));}catch {}
  }
  window.addEventListener('scroll',()=>{clearTimeout(timer);timer=setTimeout(savePosition,450);},{passive:true});
  window.addEventListener('pagehide',savePosition);
}

// Pause ambient motion when the introduction is off screen or the tab is hidden.
const heroArt = document.querySelector('.hero-art');
if (heroArt) {
  let heroVisible = true;
  const updateMotion = () => heroArt.classList.toggle('motion-paused', document.hidden || !heroVisible);
  if ('IntersectionObserver' in window) new IntersectionObserver(entries => { heroVisible = entries[0].isIntersecting; updateMotion(); }).observe(heroArt);
  document.addEventListener('visibilitychange', updateMotion);
  updateMotion();
}

// Animate once on arrival; content stays visible without JavaScript or observers.
const arrivalPreference = window.matchMedia('(prefers-reduced-motion: reduce)');
if ('IntersectionObserver' in window && !arrivalPreference.matches) {
  const arrivals = new IntersectionObserver(entries => {
    entries.forEach(entry => {
      if (!entry.isIntersecting) return;
      if (!arrivalPreference.matches) entry.target.classList.add('reveal-arrive');
      arrivals.unobserve(entry.target);
    });
  }, {threshold: 0.08});
  document.querySelectorAll('.home .map-step,.home .article-card,.home .recommend-link,.home .skill-card').forEach(el => {
    el.addEventListener('animationend', () => el.classList.remove('reveal-arrive'), {once: true});
    arrivals.observe(el);
  });
  arrivalPreference.addEventListener('change', event => {
    if (event.matches) {
      arrivals.disconnect();
      document.querySelectorAll('.reveal-arrive').forEach(el => el.classList.remove('reveal-arrive'));
    }
  });
}

