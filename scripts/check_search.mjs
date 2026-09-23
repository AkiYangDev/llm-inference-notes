import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {runInNewContext} from 'node:vm';
import {rankSearchResults} from '../site/assets/search.mjs';

const articles = JSON.parse(readFileSync(new URL('../_site/search.json', import.meta.url), 'utf8'));
for (const query of ['KV Cache', 'SGLang TP', 'DSpark', '不存在的关键词987xyz', '']) {
  const found = rankSearchResults(articles, query);
  assert.equal(new Set(found.map(r => r.article.url)).size, found.length, 'Duplicate document in results');
  for (const result of found) {
    assert.ok(articles.includes(result.article));
    if (result.anchor) assert.ok(result.article.sections.some(s => s.anchor === result.anchor));
  }
}
assert.equal(rankSearchResults(articles, '不存在的关键词987xyz').length, 0);
assert.equal(rankSearchResults(articles, '').length, articles.length);
assert.deepEqual(rankSearchResults(articles, 'DSpark'), rankSearchResults(articles, 'dspark'));
assert.ok(rankSearchResults(articles, 'SGLang TP').length > 0, 'Multi-term search should find content');
const titleHit = {title:'KV Cache', topic:'test', tags:[], excerpt:'Intro', url:'/exact/', sections:[]};
const longArticle = {title:'A long article', topic:'test', tags:[], excerpt:'Intro', url:'/long/', sections:Array.from({length:30}, (_, i) => ({heading:'KV Cache internals', text:'details', anchor:`section-${i}`}))};
const ranked = rankSearchResults([longArticle, titleHit], 'KV Cache');
assert.equal(ranked.length, 2);
assert.equal(ranked[0].article, titleHit, 'Exact title should outrank section matches');
assert.equal(ranked[1].anchor, 'section-0', 'Keep the best section deep link');
console.log(`Search checks passed: ${articles.length} documents, unique results, multi-term matching and section ranking.`);

const redirects = JSON.parse(readFileSync(new URL('../site/redirects.json', import.meta.url), 'utf8'));
for (const [oldRoute, newRoute] of Object.entries(redirects)) {
  const page = readFileSync(new URL(`../_site/${oldRoute}index.html`, import.meta.url), 'utf8');
  const script = page.match(/<script>([\s\S]*?)<\/script>/)[1];
  let destination;
  runInNewContext(script, {location: {search:'?from=bookmark', hash:'#section-2', replace: value => {destination = value;}}});
  assert.equal(destination, `/llm-inference-notes/${newRoute}?from=bookmark#section-2`);
}
console.log(`Redirect checks passed: ${Object.keys(redirects).length} legacy URLs preserve query and fragment.`);
