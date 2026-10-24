// Render formulas and code once at publication time; diagrams use a local bundle.
import {readFile, writeFile, mkdir, cp, readdir} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import {join} from 'node:path';
import {createHash} from 'node:crypto';
import {build} from 'esbuild';
import {load} from 'cheerio';
import katex from 'katex';
import hljs from 'highlight.js/lib/common';

const root = fileURLToPath(new URL('../', import.meta.url));
const out = join(root, '_site');
const vendor = join(out, 'assets/vendor');
await mkdir(vendor, {recursive:true});
await build({entryPoints:[join(root, 'node_modules/mermaid/dist/mermaid.esm.min.mjs')], bundle:true, splitting:true, format:'esm', minify:true, outdir:vendor, entryNames:'mermaid', chunkNames:'chunks/[name]-[hash]', assetNames:'[name]-[hash]', logLevel:'warning'});
await cp(join(root, 'node_modules/katex/dist/katex.min.css'), join(vendor, 'katex.min.css'));
await cp(join(root, 'node_modules/katex/dist/fonts'), join(vendor, 'fonts'), {recursive:true});
for (const [name, license] of [['mermaid','LICENSE'], ['katex','LICENSE'], ['highlight.js','LICENSE']]) {
  await cp(join(root, `node_modules/${name}/${license}`), join(vendor, `${name}-LICENSE.txt`));
}
let formulas = 0, blocks = 0;
async function enhance(folder) {
  for (const entry of await readdir(folder, {withFileTypes:true})) {
    const path = join(folder, entry.name);
    if (entry.isDirectory()) { await enhance(path); continue; }
    if (!entry.name.endsWith('.html')) continue;
    const source = await readFile(path, 'utf8');
    if (!source.includes('class="prose"')) continue;
    const $ = load(source, {decodeEntities:false});
    $('.prose pre code').each((_, element) => {
      const code = $(element);
      const language = (code.attr('class') || '').match(/language-([\w+-]+)/)?.[1];
      if (!language || !hljs.getLanguage(language)) return;
      code.html(hljs.highlight(code.text(), {language, ignoreIllegals:true}).value).addClass('hljs');
      blocks++;
    });
    $('.math-block, .math-inline').each((_, element) => {
      const node = $(element);
      // Keep the exact TeX visible if a document uses an unsupported command.
      try {
        const tex = node.attr('data-tex') ?? node.text();
        node.html(katex.renderToString(tex, {displayMode:node.hasClass('math-block'), throwOnError:true, strict:'ignore', trust:false, output:'htmlAndMathml'}));
        node.addClass('math-rendered');
        formulas++;
      } catch (error) { console.warn(`Formula kept as TeX in ${path}: ${error.message}`); }
    });
    const version = createHash('sha256').update(await readFile(join(root,'pnpm-lock.yaml'))).digest('hex').slice(0,12);
    $('head').append(`<link rel="stylesheet" href="/llm-inference-notes/assets/vendor/katex.min.css?v=${version}">`);
    await writeFile(path, $.html(), 'utf8');
  }
}
await enhance(join(out,'articles'));
await enhance(join(out,'skills'));
console.log(`Static rendering: ${blocks} code blocks, ${formulas} formulas; local Mermaid bundle ready.`);
