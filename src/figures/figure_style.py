# 本程序及代码是在人工智能工具辅助下完成的
# 人工智能工具名称：DeepSeek，型号：DeepSeek V4.1 Flash，开发公司：杭州深度求索人工智能基础技术研究有限公司，版本颁布日期：2026-09-10

"""Final cleanup for standalone figures whose captions live in the paper."""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib as mpl
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap

BLUE = "#394D96"
TEAL = "#148C84"
GOLD = "#CE9A43"
RED = "#C15C68"
INK = "#26344E"
PALE = "#E6EAF0"
NEUTRAL = "#9CADB8"
FONT_FAMILIES = ["Times New Roman", "SimSun"]


def mixed_font(**kwargs):
    """Latin/digits use Times; only missing Chinese glyphs fall back to SimSun."""
    return font_manager.FontProperties(family=FONT_FAMILIES, **kwargs)


def diverging_cmap():
    return LinearSegmentedColormap.from_list("paper-diverging", [RED, "#F5F7FA", TEAL])


def sequential_cmap():
    return LinearSegmentedColormap.from_list("paper-sequential", [PALE, TEAL, BLUE])


def format_percent4(fraction: float) -> str:
    """Format a probability as a percentage with four significant digits."""
    return f"{fraction * 100:#.4g}%"


def use_serif(root: Path):
    """Use the template's Times and SimSun fonts for all figure text."""
    times = root / "paper/fonts/Times.TTF"
    simsun = root / "paper/fonts/SimSun.ttf"
    for path in (times, simsun, root / "paper/fonts/Timesbd.TTF",
                 root / "paper/fonts/Timesi.TTF", root / "paper/fonts/Timesbi.TTF"):
        if path.is_file():
            font_manager.fontManager.addfont(str(path))
    # The electronic supplement can use the same fonts installed on the host.
    for family in FONT_FAMILIES:
        try:
            font_manager.findfont(family, fallback_to_default=False)
        except ValueError as exc:
            raise RuntimeError('绘图需要Times New Roman与SimSun；请安装字体或放入paper/fonts，见src/figures/README.md。') from exc
    mpl.rcParams["font.family"] = FONT_FAMILIES
    mpl.rcParams["mathtext.fontset"] = "custom"
    mpl.rcParams["mathtext.rm"] = "Times New Roman"
    mpl.rcParams["mathtext.it"] = "Times New Roman:italic"
    mpl.rcParams["mathtext.bf"] = "Times New Roman:bold"
    mpl.rcParams["mathtext.bfit"] = "Times New Roman:italic:bold"
    mpl.rcParams["mathtext.sf"] = "Times New Roman"
    mpl.rcParams["mathtext.tt"] = "Times New Roman"
    mpl.rcParams["mathtext.cal"] = "Times New Roman"
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["axes.unicode_minus"] = False
    mpl.rcParams["axes.spines.top"] = False
    mpl.rcParams["axes.spines.right"] = False
    mpl.rcParams["figure.facecolor"] = "white"
    mpl.rcParams["savefig.facecolor"] = "white"
    mpl.rcParams["axes.labelcolor"] = "#000000"
    mpl.rcParams["xtick.color"] = "#000000"
    mpl.rcParams["ytick.color"] = "#000000"


def black_axis_text(fig):
    """Keep axis labels, tick labels and colorbar text pure black."""
    from matplotlib.text import Text
    for text in fig.findobj(match=Text):
        props = text.get_fontproperties().copy()
        props.set_file(None)
        props.set_family(FONT_FAMILIES)
        text.set_fontproperties(props)
    def clean(ax):
        for axis in (ax.xaxis, ax.yaxis):
            axis.label.set_color("#000000")
            axis.get_offset_text().set_color("#000000")
        ax.tick_params(axis="both", which="both", labelcolor="#000000")
        for child in ax.child_axes:
            clean(child)
    for ax in fig.axes:
        clean(ax)


def prepare(fig):
    black_axis_text(fig)
    if fig._suptitle is not None:
        fig._suptitle.remove()
        fig._suptitle = None
    for ax in fig.axes:
        ax.set_title("")
        # Keep axes concise; explanations and scope belong in the paper caption.
        for getter, setter in ((ax.get_xlabel, ax.set_xlabel), (ax.get_ylabel, ax.set_ylabel)):
            value = getter()
            value = re.sub(r"（[^）]*）|\([^)]*\)", "", value).strip()
            setter(value)


def discrete_heatmap(ax, values, *, aspect="auto", **kwargs):
    """Draw constant-color vector cells; avoid viewer-specific bitmap smoothing."""
    import numpy as np
    values = np.ma.masked_invalid(np.asarray(values, dtype=float))
    rows, columns = values.shape
    mesh = ax.pcolormesh(np.arange(columns + 1) - .5,
                         np.arange(rows + 1) - .5, values,
                         shading="flat", antialiased=False, edgecolors="face",
                         linewidth=0, rasterized=False, **kwargs)
    ax.set_xlim(-.5, columns - .5)
    ax.set_ylim(rows - .5, -.5)
    ax.set_aspect(aspect)
    return mesh
