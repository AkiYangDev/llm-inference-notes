"""Preserve display TeX before Markdown interprets escapes and underscores."""
import html
import re
from markdown.extensions import Extension
from markdown.preprocessors import Preprocessor


class DisplayMath(Preprocessor):
    def run(self, lines):
        # Fenced code has already been stashed by Markdown (priority 25).
        source = '\n'.join(lines)
        def preserve(match):
            tex = html.escape(match.group(1).strip())
            return '\n' + self.md.htmlStash.store(f'<div class="math-block">{tex}</div>') + '\n'
        for pattern in (r'^\$\$[ \t]*\n(.*?)\n\$\$[ \t]*$', r'^\\\[[ \t]*\n(.*?)\n\\\][ \t]*$'):
            source = re.sub(pattern, preserve, source, flags=re.M | re.S)
        return source.split('\n')


class MathBlocksExtension(Extension):
    def extendMarkdown(self, md):
        md.preprocessors.register(DisplayMath(md), 'display_math', 24)
