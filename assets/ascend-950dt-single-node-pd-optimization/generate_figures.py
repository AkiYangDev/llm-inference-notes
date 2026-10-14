"""Regenerate the article figures. Requires matplotlib and a CJK font.

Run from any directory. Set CJK_FONT to a font file when automatic lookup fails.
The diagrams are conceptual; only latency-budget.svg contains measured data.
"""

from html import escape
from pathlib import Path
import os

OUT = Path(__file__).resolve().parent
INK, MUTED = '#18334b', '#536c82'
BLUE, TEAL, ORANGE = '#2878d0', '#087f83', '#b86427'


def text(x, y, value, size=22, color=INK, weight=400, anchor='start'):
    return (f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" '
            f'font-weight="{weight}" text-anchor="{anchor}">{escape(value)}</text>')


def rect(x, y, w, h, fill='#fff', stroke='#cbd9e4', dash=False, radius=14):
    d = ' stroke-dasharray="7 6"' if dash else ''
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="2"{d}/>')


def arrow(points, color=BLUE, dash=False):
    d = ' stroke-dasharray="7 6"' if dash else ''
    marker = 'teal' if color == TEAL else 'blue'
    return (f'<path d="{points}" fill="none" stroke="{color}" stroke-width="3" '
            f'marker-end="url(#{marker})"{d}/>')


def save(name, height, title, desc, body):
    defs = ''.join(f'<marker id="{name}" viewBox="0 0 10 10" refX="9" refY="5" '
                   f'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
                   f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{color}"/></marker>'
                   for name, color in [('blue', BLUE), ('teal', TEAL)])
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="960" height="{height}" '
           f'viewBox="0 0 960 {height}" role="img" aria-labelledby="title desc">'
           f'<title id="title">{escape(title)}</title><desc id="desc">{escape(desc)}</desc>'
           f'<defs>{defs}</defs><rect width="960" height="{height}" rx="20" fill="#f5f9fc"/>'
           '<g font-family="Noto Sans CJK SC, Microsoft YaHei, PingFang SC, sans-serif">'
           + ''.join(body) + '</g></svg>\n')
    (OUT / name).write_text(svg, encoding='utf-8')


b = [text(40, 48, '同一台机器，三种不同的职责', 28, weight=700),
     text(40, 82, '逻辑分工示意 · 请求协调、设备复用与 P→D 交接分别观察', 19, MUTED),
     rect(355, 115, 250, 76, '#e8f1fc', BLUE), text(480, 148, 'Router', 25, BLUE, 700, 'middle'),
     text(480, 176, '接收请求与协调两端', 18, MUTED, anchor='middle'),
     arrow('M 395 191 V 213 H 250 V 249'), arrow('M 565 191 V 213 H 710 V 249'),
     text(145, 235, '请求协调', 17, BLUE), text(605, 235, '请求协调', 17, BLUE),
     rect(65, 250, 355, 110, '#e8f1fc', BLUE),
     text(242, 287, 'P · Prefill', 26, BLUE, 700, 'middle'),
     text(242, 322, '处理新输入，准备缓存状态', 20, anchor='middle'),
     rect(540, 250, 355, 110, '#e2f5f2', TEAL),
     text(717, 287, 'D · Decode', 26, TEAL, 700, 'middle'),
     text(717, 322, '接收状态，推进后续生成', 20, anchor='middle'),
     arrow('M 425 307 H 534', TEAL), text(480, 284, '状态交接', 17, TEAL, anchor='middle'),
     arrow('M 240 438 V 369'), text(260, 410, '设备前缀复用', 18, BLUE),
     rect(65, 440, 355, 80, '#e8f1fc', BLUE),
     text(242, 476, '设备 KV 缓存 · 保留', 23, BLUE, 700, 'middle'),
     text(242, 505, '命中前缀继续复用', 18, MUTED, anchor='middle'),
     arrow('M 717 367 V 433', TEAL),
     rect(540, 440, 355, 80, '#e2f5f2', TEAL),
     text(717, 475, 'DSPARK 持续生成', 23, TEAL, 700, 'middle'),
     text(717, 505, 'draft → verify → 提交', 18, MUTED, anchor='middle'),
     '<path d="M 240 520 V 563" stroke="#a7b4be" stroke-width="2" stroke-dasharray="6 5"/>',
     '<path d="M 230 537 L 250 555 M 250 537 L 230 555" stroke="#b86427" stroke-width="3"/>',
     rect(65, 575, 355, 77, '#fff3e9', '#cd9873', True),
     text(242, 607, 'HiCache 主机缓存 · 关闭', 22, ORANGE, 700, 'middle'),
     text(242, 636, '移除额外缓存层的路径', 18, MUTED, anchor='middle'),
     text(540, 594, 'HiCache 关闭后，', 21, weight=700),
     text(540, 629, '必要的 P→D 交接仍然保留。', 21, weight=700),
     text(40, 697, '单机 PD 不等于没有交接；关闭主机缓存也不等于关闭前缀复用。', 18, MUTED)]
save('pd-cache-paths.svg', 730, '单机 PD 与缓存路径',
     'P 和 D 分别负责输入处理与生成，设备前缀缓存保留；HiCache 主机缓存路径关闭，P 到 D 的状态交接继续。', b)

b = [text(40, 48, '一次验证，确定可以向前推进多远', 28, weight=700),
     text(40, 82, '概念示意 · 6 个候选中前 4 个通过，不是本次实测接受长度', 19, MUTED)]
for y, label, sub in [(140, 'Draft', '准备候选'), (275, 'Verify', '验证可提交前缀'), (420, 'Commit', '保留有效结果')]:
    b += [text(45, y+28, label, 25, weight=700), text(45, y+59, sub, 18, MUTED)]
for i in range(6):
    x = 268+i*94
    b += [rect(x, 140, 76, 64, '#e8f1fc', BLUE, radius=10),
          text(x+38, 181, f't{i+1}', 25, BLUE, 700, 'middle')]
    color, fill = (TEAL, '#e2f5f2') if i < 4 else ((ORANGE, '#fff3e9') if i == 4 else (MUTED, '#e9eef2'))
    b += [rect(x, 275, 76, 64, fill, color, radius=10),
          text(x+38, 315, '通过' if i < 4 else ('拒绝' if i == 4 else '后缀'), 20, color, 700, 'middle')]
    if i < 4:
        b += [rect(x, 420, 76, 64, '#e2f5f2', TEAL, radius=10),
              text(x+38, 461, f't{i+1}', 25, TEAL, 700, 'middle')]
b += [arrow('M 538 211 V 263'), arrow('M 447 346 V 406', TEAL),
      text(738, 377, '拒绝位置及后缀不提交', 18, ORANGE, anchor='middle'),
      text(665, 455, '从接受边界继续', 21, TEAL, 700),
      rect(40, 525, 880, 64, '#fff', '#d9e4ed'),
      text(480, 565, '有效产出看“提交数 / 周期时间”，不能只看候选数。', 23, INK, 700, 'middle'),
      text(40, 626, '为突出接受边界，图中省略修正或额外 token 的处理。', 18, MUTED)]
save('speculative-cycle.svg', 656, '投机解码的候选验证与提交',
     '示意6个候选，前4个通过，第5个拒绝，后缀不提交。提交的有效token数与周期时间共同决定收益。', b)

b = [text(40, 48, '图档位更接近实际 batch，可以少填充一些槽位', 27, weight=700),
     text(40, 82, '示例：实际 BS = 5，runner 选择不小于它的最近捕获档位', 19, MUTED),
     text(40, 166, '旧档位', 24, weight=700), text(40, 197, '1 / 2 / 4 / 16', 19, MUTED),
     text(40, 296, '新档位', 24, weight=700), text(40, 327, '加入 6 / 8 / …', 19, MUTED)]
for row, n in [(145,16),(275,6)]:
    for i in range(n):
        b += [rect(265+i*38, row, 30, 54, '#2878d0' if i<5 else '#dbe5ed',
                   '#2878d0' if i<5 else '#c3d0db', radius=5)]
b += [text(265, 232, '捕获 BS 16：5 个真实请求 + 11 个填充槽位', 20, MUTED),
      text(265, 362, '捕获 BS 6：5 个真实请求 + 1 个填充槽位', 20, TEAL, 700),
      rect(40, 400, 880, 69, '#fff', '#d9e4ed'),
      text(480, 442, '图的最大 BS ≠ 客户端并发上限；本地 batch 还受 DP 与执行阶段影响。', 20, anchor='middle'),
      text(40, 509, '蓝色：真实请求槽位    灰色：padding    ·    槽位减少比例不是实测加速比', 18, MUTED)]
save('graph-padding.svg', 540, '图捕获档位与padding',
     '在最近向上档位假设下，实际BS5在旧档位补到16，新档位补到6，填充槽位从11降到1。', b)


def latency_chart():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    font_path = os.environ.get('CJK_FONT')
    if font_path:
        font_manager.fontManager.addfont(font_path)
        family = font_manager.FontProperties(fname=font_path).get_name()
    else:
        family = 'Noto Sans CJK SC'
        font_manager.findfont(family, fallback_to_default=False)
    plt.rcParams.update({'font.family': family, 'svg.fonttype': 'path', 'font.size': 13})
    ttft, e2e = 12.03773, 15.66995
    tail = e2e - ttft
    fig, ax = plt.subplots(figsize=(10, 4.8), facecolor='#f5f9fc')
    fig.subplots_adjust(left=.07, right=.96, bottom=.32, top=.66)
    ax.set_facecolor('#f5f9fc')
    ax.barh([0], [ttft], height=.46, color=BLUE)
    ax.barh([0], [tail], left=[ttft], height=.46, color=TEAL)
    for center, label in [(ttft/2, '首 token 前\n12.04 秒 · 76.8%'),
                          (ttft+tail/2, '首 token 后\n3.63 秒 · 23.2%')]:
        ax.text(center, 0, label, ha='center', va='center', color='white', fontsize=14, linespacing=1.65)
    ax.set_xlim(0, 16)
    ax.set_ylim(-.5, .5)
    ax.set_yticks([])
    ax.set_xticks([0,4,8,12,16], ['0', '4', '8', '12', '16 秒'])
    ax.tick_params(axis='x', colors=MUTED, length=0, pad=12)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.text(.055, .89, '平均 15.67 秒里，12.04 秒发生在首 token 之前', fontsize=19, weight='bold', color=INK)
    fig.text(.055, .80, 'N192 / C48 / 2 req/s · 128K/1K 场景 · 实际 Cache hit 91.3%', fontsize=12, color=MUTED)
    fig.text(.055, .19, '首 token 后 = Mean E2E − Mean TTFT = 3.63222 秒', fontsize=13, color=INK)
    fig.text(.055, .09, '实测均值分解：蓝色部分尚未拆分为排队、P 计算或 KV 交接。', fontsize=12, color=MUTED)
    fig.savefig(OUT / 'latency-budget.svg', metadata={'Date': None, 'Creator': 'AkiYangDev / matplotlib'})
    plt.close(fig)


latency_chart()
print('Generated four figures in', OUT)
