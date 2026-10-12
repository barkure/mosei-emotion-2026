# 本程序及代码是在人工智能工具辅助下完成的
# 人工智能工具名称：DeepSeek，型号：DeepSeek V4.1 Flash，开发公司：杭州深度求索人工智能基础技术研究有限公司，版本颁布日期：2026-09-10

"""Draw the chapter-three data overview from the frozen attachment-two audit."""
from __future__ import annotations
from figure_style import mixed_font

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np

from figure_style import BLUE, TEAL, INK, PALE, NEUTRAL, prepare, use_serif

ROOT = Path(__file__).resolve().parents[2]
AUDIT = json.loads((ROOT / "outputs/data_audit/attachment2_audit.json").read_text(encoding="utf-8"))
OUT = ROOT / "paper/figures/数据概览.pdf"
CJK = mixed_font()
NEGATIVE = BLUE
POSITIVE = TEAL
UNOBSERVED = PALE

use_serif(ROOT)
plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42, "axes.unicode_minus": False,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.spines.left": False, "axes.spines.bottom": False})


def label_counts(part: dict) -> np.ndarray:
    return np.array([part["labels"]["classification_counts"][f"{i}.0"] for i in range(3)], dtype=float)


def style_axis(ax, labels: list[str]) -> None:
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100], ["0", "25", "50", "75", "100%"])
    ax.tick_params(axis="x", length=0, colors=INK, labelsize=9, pad=7)
    ax.tick_params(axis="y", length=0, pad=12)
    ax.set_yticks([1, 0], labels, fontproperties=CJK)
    ax.set_ylim(-0.55, 1.55)
    for x in [0, 25, 50, 75, 100]:
        ax.axvline(x, color="#E9EDF0", lw=0.7, zorder=0)


def stacked_bar(ax, y: float, values: np.ndarray, colors: list[str],
                dark_text: set[int] | None = None, show_indices: set[int] | None = None) -> None:
    dark_text = dark_text or set()
    start = 0.0
    for i, (value, color) in enumerate(zip(values, colors)):
        ax.barh(y, value, left=start, height=0.38, color=color, edgecolor="white", linewidth=0.6, zorder=2)
        if value >= 11 and (show_indices is None or i in show_indices):
            ax.text(start + value / 2, y, f"{value:.3f}%" if value < 10 else f"{value:.2f}%", ha="center", va="center",
                    color=INK if i in dark_text else "white", fontsize=9.5, zorder=3)
        start += value


def main() -> None:
    splits = AUDIT["versions"]["aligned"]["splits"]
    train, valid = splits["train"], splits["valid"]
    train_pct = label_counts(train) / label_counts(train).sum() * 100
    valid_pct = label_counts(valid) / label_counts(valid).sum() * 100
    audio = train["modalities"]["audio"]
    vision = train["modalities"]["vision"]
    audio_pct = 100 * audio["observed_positions"] / audio["struct_positions"]
    vision_pct = 100 * vision["observed_positions"] / vision["struct_positions"]

    fig, axes = plt.subplots(2, 1, figsize=(6.6, 3.1))
    fig.subplots_adjust(left=0.15, right=0.985, top=0.91, bottom=0.12, hspace=0.88)
    upper, lower = axes
    style_axis(upper, ["训练集", "验证集"])
    stacked_bar(upper, 1, train_pct, [NEGATIVE, NEUTRAL, POSITIVE], {1})
    stacked_bar(upper, 0, valid_pct, [NEGATIVE, NEUTRAL, POSITIVE], {1})
    upper.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in [NEGATIVE, NEUTRAL, POSITIVE]],
                 labels=["负向", "中性", "正向"], ncol=3, loc="lower left", bbox_to_anchor=(0, 1.04),
                 frameon=False, prop=CJK, handlelength=1.2, columnspacing=1.8, borderaxespad=0)

    style_axis(lower, ["语音", "视频"])
    stacked_bar(lower, 1, np.array([audio_pct, 100 - audio_pct]), [NEGATIVE, UNOBSERVED], show_indices={0})
    stacked_bar(lower, 0, np.array([vision_pct, 100 - vision_pct]), [NEGATIVE, UNOBSERVED], show_indices={0})
    for y, pct in [(1, audio_pct), (0, vision_pct)]:
        remainder = 100 - pct
        label = f"{remainder:.3f}%" if remainder < 10 else f"{remainder:.2f}%"
        lower.text(pct + remainder / 2, y, label, ha="center", va="center", fontsize=9, color=INK)
    lower.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in [NEGATIVE, UNOBSERVED]],
                 labels=["有观测", "未观测"], ncol=2, loc="lower left", bbox_to_anchor=(0, 1.04),
                 frameon=False, prop=CJK, handlelength=1.2, columnspacing=1.8, borderaxespad=0)
    prepare(fig)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
