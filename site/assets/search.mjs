// Keep one best destination per document so long articles cannot crowd out others.
export function rankSearchResults(articles, query) {
  const needle = query.trim().toLowerCase();
  const terms = needle.split(/\s+/).filter(Boolean);
  const matches = value => terms.every(term => value.toLowerCase().includes(term));
  const found = [];
  for (const article of articles) {
    const meta = [article.title, article.topic, ...(article.tags || []), article.excerpt].join(' ');
    let best = !needle || matches(meta)
      ? {article, heading: '', text: article.excerpt, anchor: '', score: !needle ? 0 : article.title.toLowerCase().includes(needle) ? 5 : 3}
      : null;
    if (needle) for (const section of article.sections || []) {
      if (!matches(meta + ' ' + section.heading + ' ' + section.text)) continue;
      const score = matches(section.heading) ? 4 : 1;
      if (!best || score > best.score) best = {article, ...section, score};
    }
    if (best) found.push(best);
  }
  return found.sort((a, b) => b.score - a.score);
}
