"""Small editorial helpers shared by discovery and long-form reading pages."""
import html
import math
import re


def reading_minutes(source):
    """An estimate including code/diagram study, not a measured reading speed."""
    fences = re.findall(r'(```|~~~)[^\n]*\n([\s\S]*?)\1', source)
    code_lines = sum(len(body.strip().splitlines()) for _, body in fences)
    diagrams = len(re.findall(r'^(?:```|~~~)mermaid\b', source, re.M))
    prose = re.sub(r'(```|~~~)[\s\S]*?\1', '', source)
    prose = re.sub(r'\[([^]]+)\]\([^)]*\)', r'\1', prose)
    prose = re.sub(r'<[^>]+>|https?://\S+', '', prose)
    chinese = len(re.findall(r'[\u3400-\u9fff]', prose))
    words = len(re.findall(r'[A-Za-z0-9_]+', prose))
    return max(1, math.ceil(chinese / 400 + words / 180 + code_lines / 35 + diagrams * .5))


def source_status(source, dates):
    intro = re.split(r'^## ', source, maxsplit=1, flags=re.M)[0]
    checked = re.search(r'(?:核对日期|核对时间)[^\d\n]{0,10}(20\d{2}-\d{2}-\d{2})', intro)
    stamp = dates.get('modified', '')
    bits = []
    if stamp:
        bits.append(f'<time datetime="{html.escape(stamp)}">更新于 {stamp[:10]}</time>')
    if checked:
        bits.append('源码核对 ' + checked.group(1))
    return '<div class="source-status">' + '<span> · </span>'.join(bits) + '</div>' if bits else ''


def mobile_directory(toc):
    if not toc:
        return ''
    return f'''<button class="toc-fab" aria-controls="toc-dialog" aria-haspopup="dialog" hidden>本页目录</button><dialog id="toc-dialog" aria-labelledby="toc-dialog-title"><div class="dialog-top"><h2 id="toc-dialog-title">本页目录</h2><button class="close-toc icon-button" aria-label="关闭目录">×</button></div>{toc}</dialog>'''
