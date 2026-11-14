# 读取原始 EPLB NPZ

原始数据来自 2026-10-31 的历史采集。两组是独立执行，不能按行当成同一个 token 的对照。

在解压后的目录安装并执行：

```bash
python -m pip install numpy==2.4.1 matplotlib==3.10.8
python examples/eplb-count-npz-reading/read_counts.py --output count-summary.json
python examples/eplb-count-npz-reading/plot_counts.py --output window-heatmap.png
```

两个 NPZ 的 SHA256 在读取脚本中核对。计数已经对 16 个 rank 求和。
画图保留第 0 层前 32 条 append 和所有 272 个物理槽位，共同线性色标，无插值。
图片是数据图，不是终端截图。图中文字使用英文，以便不同系统无需额外中文字体。
