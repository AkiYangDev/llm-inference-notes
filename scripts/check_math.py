"""Exercise TeX preservation and fenced-code isolation at the Markdown boundary."""
import sys
from pathlib import Path
import markdown

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'site'))
from math_blocks import MathBlocksExtension

def render(text):
    return markdown.markdown(text, extensions=['fenced_code', MathBlocksExtension()])

tex = r'\alpha(x)=\min\left(1,\frac{p_i(x)}{q_i(x)}\right)'
assert f'<div class="math-block">{tex}</div>' in render('$$\n' + tex + '\n$$')
assert f'<div class="math-block">{tex}</div>' in render('\\[\n' + tex + '\n\\]')
assert 'p_i' in render('$$\np_i < q_i\n$$') and '&lt;' in render('$$\np_i < q_i\n$$')
for fence in ('```', '~~~'):
    output = render(f'{fence}text\n$$\nx_y\n$$\n{fence}')
    assert 'math-block' not in output and '<pre><code' in output
assert 'math-block' not in render('An unmatched delimiter:\n\n$$\nx_y')
assert render('$$\nx\n$$\n\n$$\ny\n$$').count('class="math-block"') == 2
print('Math checks passed: TeX preserved, HTML escaped and fenced examples excluded.')
