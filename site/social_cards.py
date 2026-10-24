"""Generate readable article-specific PNG cards without a browser or remote fonts."""
from pathlib import Path
from hashlib import sha256
from PIL import Image, ImageDraw, ImageFont


def font_path():
    candidates = [
        Path('C:/Windows/Fonts/msyh.ttc'),
        Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'),
        Path('/System/Library/Fonts/PingFang.ttc'),
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    raise RuntimeError('Social cards need a Chinese font: install fonts-noto-cjk on Linux.')


def wrapped(draw, text, font, width, limit):
    lines, line = [], ''
    for char in text:
        if line and draw.textlength(line + char, font=font) > width:
            lines.append(line)
            line = ''
        line += char
    if line:
        lines.append(line)
    if len(lines) > limit:
        lines = lines[:limit]
        while draw.textlength(lines[-1] + '…', font=font) > width:
            lines[-1] = lines[-1][:-1]
        lines[-1] += '…'
    return lines


def make_card(out, slug, title, topic, summary):
    image = Image.new('RGB', (1200, 630), '#f8fafd')
    draw = ImageDraw.Draw(image)
    font = font_path()
    small = ImageFont.truetype(font, 26)
    heading = ImageFont.truetype(font, 49)
    body = ImageFont.truetype(font, 27)
    draw.rectangle((0, 0, 15, 630), fill='#235fd3')
    draw.rounded_rectangle((1010, -95, 1330, 225), radius=90, fill='#e3ecfc')
    draw.text((70, 48), topic + ' / 推理工程手记', font=small, fill='#235fd3')
    for i, line in enumerate(wrapped(draw, title, heading, 1050, 3)):
        draw.text((70, 120 + i * 72), line, font=heading, fill='#182c46')
    for i, line in enumerate(wrapped(draw, summary, body, 1050, 2)):
        draw.text((70, 374 + i * 43), line, font=body, fill='#52677d')
    draw.line((70, 518, 1130, 518), fill='#dfe8f3', width=2)
    draw.text((70, 551), 'AkiYang', font=small, fill='#182c46')
    draw.text((760, 554), 'SGLang · Ascend · DeepSeek', font=small, fill='#52677d')
    version = sha256((title + topic + summary).encode('utf-8')).hexdigest()[:10]
    relative = 'assets/cards/' + slug.replace('/', '--') + '-' + version + '.png'
    target = out / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    image.save(target, optimize=True)
    return relative
