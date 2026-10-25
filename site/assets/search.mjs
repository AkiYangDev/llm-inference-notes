// Spelling variants and equivalent terms only: AL is not acceptance rate,
// and TPOT is not ITL.
const aliases = [
  ['ttft', ['ttft', '首字延迟', '首 token 延迟', '首token延迟', 'time to first token']],
  ['tpot', ['tpot', '每 token 耗时', '每token耗时', 'time per output token']],
  ['itl', ['itl', 'token 间延迟', 'token间延迟', 'inter token latency']],
  ['kvcache', ['kv cache', 'kvcache', 'kv_cache', 'kv-cache', 'kv 缓存', 'kv缓存']],
  ['pddisaggregation', ['pd 分离', 'pd分离', 'prefill decode disaggregation', 'prefill/decode disaggregation', 'pd disaggregation']],
  ['alltoall', ['all-to-all', 'alltoall', 'all_to_all', 'all to all']],
  ['tensorparallel', ['tensor parallel', '张量并行']],
  ['expertparallel', ['expert parallel', '专家并行']],
  ['speculative', ['speculative decoding', '投机解码']],
  ['prefixcache', ['prefix cache', 'prefixcache', '前缀缓存']],
];
const escape = value => value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
const variants = aliases.flatMap(([canonical, names]) => names.map(name => ({canonical, name})))
  .sort((a,b) => b.name.length - a.name.length);
const aliasPattern = new RegExp(variants.map(({name}) => {
  const body = escape(name).replace(/ /g, '\\s*');
  return /^[a-z]+$/i.test(name) ? `(?<![a-z])${body}(?![a-z])` : body;
}).join('|'), 'gi');

export function normalizeSearchText(value) {
  return String(value || '').normalize('NFKC').toLowerCase().replace(aliasPattern, match => {
    const key = match.replace(/\s/g,'');
    return variants.find(v => v.name.replace(/\s/g,'') === key)?.canonical || match;
  }).replace(/([a-z0-9])[-_/](?=[a-z0-9])/g, '$1');
}
export function searchTerms(query) {
  return [...new Set(normalizeSearchText(query).trim().split(/\s+/).filter(Boolean))];
}
export function matchesQuery(value, query) {
  const text = normalizeSearchText(value);
  return searchTerms(query).every(term => text.includes(term));
}
export function highlightTerms(query) {
  const terms = query.trim().toLowerCase().split(/\s+/).filter(Boolean);
  const normalized = normalizeSearchText(query);
  for (const [canonical, names] of aliases) if (normalized.includes(canonical)) terms.push(...names);
  return [...new Set(terms)].sort((a,b) => b.length - a.length);
}

const prepared = new WeakMap();
function prepare(article) {
  if (!prepared.has(article)) prepared.set(article, {
    title:normalizeSearchText(article.title),
    meta:normalizeSearchText([article.title, article.topic, ...(article.tags || []), article.excerpt].join(' ')),
    sections:(article.sections || []).map(section => ({section, heading:normalizeSearchText(section.heading), text:normalizeSearchText(section.text)})),
  });
  return prepared.get(article);
}

// One best destination per document keeps long articles from filling results.
export function rankSearchResults(articles, query) {
  const terms = searchTerms(query), needle = normalizeSearchText(query).trim();
  const matches = value => terms.every(term => value.includes(term));
  const found = [];
  for (const article of articles) {
    const {title, meta, sections} = prepare(article);
    let best = matches(meta)
      ? {article, heading:'', text:article.excerpt, anchor:'', score:!terms.length ? 0 : title === needle ? 100 : title.includes(needle) ? 60 : 25}
      : null;
    if (terms.length) for (const {section, heading, text} of sections) {
      if (!matches(meta + ' ' + heading + ' ' + text)) continue;
      const score = matches(heading) ? 45 : 10;
      if (!best || score > best.score) best = {article, ...section, score};
    }
    if (best) found.push(best);
  }
  return found.sort((a,b) => b.score - a.score);
}
