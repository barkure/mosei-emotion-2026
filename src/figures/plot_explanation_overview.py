# 本程序及代码是在人工智能工具辅助下完成的
# 人工智能工具名称：DeepSeek，型号：DeepSeek V4.1 Flash，开发公司：杭州深度求索人工智能基础技术研究有限公司，版本颁布日期：2026-09-10

"""Validation-set Shapley distributions for the problem-three paper section."""
from __future__ import annotations
from figure_style import mixed_font, discrete_heatmap

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.font_manager import FontProperties

from figure_style import BLUE, TEAL, GOLD, INK, PALE, diverging_cmap, prepare, sequential_cmap, use_serif

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper/figures"
FONT = mixed_font()
use_serif(ROOT)
plt.rcParams.update({"font.size": 9, "axes.unicode_minus": False})
rows = json.loads((ROOT / "outputs/q3_explanation/q3/validation_explanation_all.json").read_text(encoding="utf-8"))["details"]
assert len(rows) == 728
names = ["负向", "中性", "正向"]
mods = ["文本", "语音", "视频"]
colors = [BLUE, TEAL, GOLD]


def save(fig, name=None, gallery_name=None):
    prepare(fig)
    if name is not None:
        fig.savefig(OUT / name, bbox_inches="tight", pad_inches=.10)
    if gallery_name is not None:
        target = ROOT / "outputs/figures" / gallery_name
        fig.savefig(target, bbox_inches="tight", pad_inches=.10)
        print(target)
    plt.close(fig)
    if name is not None:
        print(OUT / name)


# Each small black segment marks the sample median; the dashed line marks zero.
rng = np.random.default_rng(2026)
fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.95), sharey=True,
                         gridspec_kw={"wspace": .12})
for k, ax in enumerate(axes):
    subset = [row for row in rows if row["polarity"] == k]
    for j, color in enumerate(colors):
        values = np.array([row["shapley_probability"][j] for row in subset])
        ax.scatter(j + rng.uniform(-.17, .17, len(values)), values,
                   s=4.5, color=color, alpha=.26, rasterized=True)
        ax.plot([j-.20, j+.20], [np.median(values)]*2, color=INK, lw=2.0,
                solid_capstyle="round", zorder=4)
    ax.axhline(0, color=INK, ls=(0, (3, 3)), lw=.75, alpha=.65)
    ax.set_xticks(range(3), mods, fontproperties=FONT)
    ax.set_xlim(-.35, 2.35)
    ax.set_xlabel(f"预测{names[k]}  {len(subset)}条", fontproperties=FONT)
    ax.tick_params(axis="y", length=3, color="black")
    ax.tick_params(axis="x", length=3, color="black")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["bottom"].set_color("black")
    ax.spines["bottom"].set_linewidth(.8)
axes[0].set_ylabel("有符号 Shapley 概率贡献", fontproperties=FONT)
fig.subplots_adjust(left=.10, right=.99, top=.97, bottom=.22)
save(fig, "预测类别模态贡献分布.pdf")


# Consecutive validation indices 0--29 illustrate the local window scan.
first = sorted(rows, key=lambda row: row["valid_index"])[:30]
assert [row["valid_index"] for row in first] == list(range(30))
windows = [row["local"]["text"]["all_window_drops"] for row in first]
width = max(map(len, windows))
values = np.full((30, width), np.nan)
for i, scores in enumerate(windows):
    values[i, :len(scores)] = scores
palette = diverging_cmap().copy()
palette.set_bad(PALE)
fig, ax = plt.subplots(figsize=(7.15, 3.65))
fig.subplots_adjust(left=.10, right=.91, bottom=.18, top=.97)
limit = float(np.ceil(np.nanmax(np.abs(values)) * 5) / 5)
im = discrete_heatmap(ax, values, aspect="auto", cmap=palette,
               norm=Normalize(vmin=-limit, vmax=limit))
ax.set_xticks([0, 9, 19, 29, 38], ["1", "10", "20", "30", "39"])
ax.set_yticks([0, 4, 9, 14, 19, 24, 29], ["1", "5", "10", "15", "20", "25", "30"])
ax.set_xlabel("文本候选窗口序号", fontproperties=FONT)
ax.set_ylabel("验证集样本序号", fontproperties=FONT)
ax.tick_params(length=0)
for spine in ax.spines.values():
    spine.set_visible(False)
cax = ax.inset_axes([1.018, 0, .032, 1])
bar = fig.colorbar(im, cax=cax, ticks=np.linspace(-limit, limit, 9))
bar.outline.set_visible(False)
bar.set_label("遮挡后预测类别概率下降", fontproperties=FONT)
save(fig, "文本局部窗口响应.pdf")
