# EPLB 专题算例

`eplb_models.py` 只使用 Python 标准库，计算负载、副本、本地复用、源覆盖、时间尺度、收益上限与传输字节。
所有时间、带宽和负载数字均为明确的教学设定，不是新的设备测试。

```bash
python examples/eplb-study-lab/eplb_models.py --output models.json
```

可选的收益上限图使用 Matplotlib：

```bash
python -m pip install matplotlib==3.10.8
python examples/eplb-study-lab/plot_bound.py --output speedup-bound.png
```

原始 NPZ 计数与相关读取、绘图脚本位于另一份 `eplb-count-npz-reading` 数据包。
`commit-index.json` 记录 PR #49701 的实际提交与后续收敛结果；不能将中间版本的尝试写成最终已保留优化。
