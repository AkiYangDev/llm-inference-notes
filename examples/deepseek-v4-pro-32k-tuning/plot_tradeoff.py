"""Render the three measured N96 points; requires matplotlib."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "assets/deepseek-v4-pro-32k-tuning"
with (ASSETS / "measurements.csv").open(newline="") as handle:
    rows = sorted(
        (row for row in csv.DictReader(handle) if row["num_prompts"] == "96"),
        key=lambda row: int(row["client_concurrency"]),
    )
labels = [f'C{row["client_concurrency"]}' for row in rows]
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False, "axes.spines.bottom": False,
    "svg.fonttype": "none",
})
figure, axes = plt.subplots(1, 2, figsize=(10, 4.5), layout="constrained")
panels = [
    ([float(row["mean_ttft_ms"]) / 1000 for row in rows], "Mean TTFT (s)", "#267ca5", 36),
    ([float(row["output_throughput_tok_s"]) for row in rows], "Output throughput (tok/s)", "#32766b", 1300),
]
for axis, (values, title, color, limit) in zip(axes, panels):
    bars = axis.bar(labels, values, width=0.55, color=color)
    axis.set_title(title, loc="left", fontsize=14, fontweight="medium", pad=18)
    axis.set_ylim(0, limit)
    axis.set_axisbelow(True)
    axis.grid(axis="y", color="#e4ebef", linewidth=0.8)
    axis.tick_params(length=0, labelcolor="#394b57", pad=8)
    axis.bar_label(bars, labels=[f"{value:,.2f}" for value in values], padding=8, color="#213744")
figure.suptitle("N=96  |  Cache hit rate: 91.8%  |  One run per point", fontsize=11, color="#516575")
figure.savefig(ASSETS / "concurrency-tradeoff.svg", metadata={"Date": None, "Creator": "Matplotlib"})
plt.close(figure)
