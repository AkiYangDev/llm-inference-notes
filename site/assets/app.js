const base = new URL('../', import.meta.url);
const theme = document.querySelector('.theme-toggle');
function themeLabel(){theme.setAttribute('aria-label',document.documentElement.dataset.theme==='dark'?'切换浅色模式':'切换深色模式');}
themeLabel();
theme.addEventListener('click',()=>{const next=document.documentElement.dataset.theme==='dark'?'light':'dark';document.documentElement.dataset.theme=next;try{localStorage.setItem('aki-theme',next)}catch{}themeLabel()});
const search=document.querySelector('#search-dialog'), input=document.querySelector('#search-input'), results=document.querySelector('#search-results');
let index;
async function searchArticles(){
  try{index ||= await fetch(new URL('search.json',base)).then(r=>{if(!r.ok)throw Error();return r.json()});
    const q=input.value.trim().toLowerCase();
    const found=index.filter(a=>(a.title+' '+a.text).toLowerCase().includes(q));
    results.replaceChildren();
    if(!found.length){results.textContent='没有找到相关文章，试试其他关键词。';return}
    for(const a of found){const link=document.createElement('a');link.href=a.url;link.className='search-result';const title=document.createElement('strong');title.textContent=a.title;const sub=document.createElement('span');const pos=a.text.toLowerCase().indexOf(q);sub.textContent=q?a.text.slice(Math.max(0,pos-30),pos+110):a.topic+' · 约 '+a.minutes+' 分钟';link.append(title,sub);results.append(link)}
  }catch{results.textContent='搜索暂时无法加载，请关闭窗口后从文章目录浏览。'}
}
function openSearch(){search.showModal();input.focus();searchArticles()}
document.querySelector('.search-trigger').addEventListener('click',openSearch);
document.querySelector('.close-search').addEventListener('click',()=>search.close());
input.addEventListener('input',searchArticles);
document.addEventListener('keydown',e=>{if(e.key==='/'&&!/INPUT|TEXTAREA|SELECT/.test(document.activeElement.tagName)&&!document.activeElement.isContentEditable&&!search.open){e.preventDefault();openSearch()}});
for(const dialog of document.querySelectorAll('dialog'))dialog.addEventListener('click',e=>{if(e.target===dialog){const r=dialog.getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)dialog.close()}});
for(const block of document.querySelectorAll('.prose pre:not(.mermaid)')){
  const button=document.createElement('button');button.className='copy-code';button.textContent='复制';button.setAttribute('aria-label','复制代码');
  button.addEventListener('click',async()=>{try{await navigator.clipboard.writeText(block.querySelector('code')?.textContent||'');button.textContent='已复制'}catch{button.textContent='请手动选择复制'}setTimeout(()=>button.textContent='复制',2000)});block.append(button);
}
if(document.querySelector('.prose')){
  const progress=document.querySelector('.reading-progress');
  const update=()=>{const doc=document.documentElement;progress.style.width=(doc.scrollTop/Math.max(1,doc.scrollHeight-doc.clientHeight)*100)+'%'};window.addEventListener('scroll',update,{passive:true});update();
  const observer=new IntersectionObserver(entries=>{for(const e of entries)if(e.isIntersecting){document.querySelectorAll('.toc a').forEach(a=>a.classList.toggle('current',decodeURIComponent(a.hash.slice(1))===e.target.id))}},{rootMargin:'-100px 0px -60% 0px'});document.querySelectorAll('.prose h2').forEach(h=>observer.observe(h));
  try{const {default:hljs}=await import('https://cdn.jsdelivr.net/npm/@highlightjs/cdn-assets@11.11.1/es/highlight.min.js');const css=document.createElement('link');css.rel='stylesheet';css.href='https://cdn.jsdelivr.net/npm/@highlightjs/cdn-assets@11.11.1/styles/github-dark.min.css';document.head.append(css);document.querySelectorAll('.prose pre code').forEach(c=>hljs.highlightElement(c))}catch{}
  const diagrams=[...document.querySelectorAll('.mermaid')];
  if(diagrams.length){try{const {default:mermaid}=await import('https://cdn.jsdelivr.net/npm/mermaid@11.4.1/dist/mermaid.esm.min.mjs');mermaid.initialize({startOnLoad:false,securityLevel:'strict',theme:'base',themeVariables:{primaryColor:'#edf2ff',primaryTextColor:'#25385b',primaryBorderColor:'#9eafe0',lineColor:'#7686aa',secondaryColor:'#f5f7fc',tertiaryColor:'#fff',fontFamily:'sans-serif',fontSize:'15px'},flowchart:{htmlLabels:false}});await mermaid.run({nodes:diagrams});
    for(const diagram of diagrams){const button=document.createElement('button');button.textContent='放大查看';button.className='expand-diagram';button.addEventListener('click',()=>{const view=document.querySelector('#diagram-view');view.replaceChildren(diagram.querySelector('svg').cloneNode(true));document.querySelector('#diagram-dialog').showModal()});diagram.parentElement.append(button)}
  }catch{for(const diagram of diagrams){const p=document.createElement('p');p.className='diagram-error';p.textContent='图表暂时无法加载，以下保留 Mermaid 源码。';diagram.before(p)}}}
}
document.querySelector('.close-diagram').addEventListener('click',()=>document.querySelector('#diagram-dialog').close());
