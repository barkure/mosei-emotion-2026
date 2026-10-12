# 本程序及代码是在人工智能工具辅助下完成的
# 人工智能工具名称：DeepSeek，型号：DeepSeek V4.1 Flash，开发公司：杭州深度求索人工智能基础技术研究有限公司，版本颁布日期：2026-09-10

"""Validation diagnostics for the gallery and the paper diagnostic figure.

No model fitting or test-set reads. Run with the existing pixi environment.
"""
from __future__ import annotations
from figure_style import mixed_font

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
from scipy.stats import gaussian_kde
from figure_style import BLUE, TEAL, GOLD, RED, INK, PALE, format_percent4, prepare, use_serif

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/figures"
OUT.mkdir(parents=True, exist_ok=True)
FONT = mixed_font()
NAMES = ["负向", "中性", "正向"]
MODS = ["文本", "语音", "视频"]
COLORS = [BLUE, TEAL, GOLD]
plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42,
                     "axes.unicode_minus": False, "axes.spines.top": False,
                     "axes.spines.right": False})
use_serif(ROOT)


def title(ax, s):
    ax.set_title(s, fontproperties=FONT, fontsize=12, color=INK, pad=13)


def save(fig, name, paper_name=None):
    prepare(fig)
    fig.savefig(OUT / name, bbox_inches="tight", pad_inches=.13)
    if paper_name is not None:
        paper_target = ROOT / "paper/figures" / paper_name
        fig.savefig(paper_target, bbox_inches="tight", pad_inches=.10)
        print(paper_target)
    plt.close(fig)
    print(OUT / name)


report = json.loads((ROOT / "outputs/q2_prediction/q2_systematic/gate_recon_ensemble.json").read_text(encoding="utf-8"))
explain = json.loads((ROOT / "outputs/q3_explanation/q3/validation_explanation_all.json").read_text(encoding="utf-8"))
assert report["data_version"] == explain["data_version"] == "aligned_50"
assert report["valid_n"] == explain["n_total_valid"] == 728
rows = explain["details"]
assert len(rows) == 728 and sorted(r["valid_index"] for r in rows) == list(range(728))
true = np.array([r["true_polarity"] for r in rows], dtype=int)
pred = np.array([r["polarity"] for r in rows], dtype=int)
probs = np.array([r["full_probabilities"] for r in rows], dtype=float)
assert np.array_equal(probs.argmax(axis=1), pred)

# 17. Reliability of the selected-class confidence, without calibration fitting.
confidence = probs.max(axis=1)
correct = pred == true
edges = np.linspace(0, 1, 11)
bins = np.minimum(np.digitize(confidence, edges) - 1, 9)
fig, ax = plt.subplots(figsize=(7.2, 3.65))
ax.plot([0, 1], [0, 1], ls="--", color="#9AA6BA", lw=1.4)
for b in range(10):
    mask = bins == b
    if mask.sum() == 0:
        continue
    x = confidence[mask].mean()
    p = correct[mask].mean()
    n = mask.sum()
    # Wilson 95% interval; descriptive uncertainty for a fixed validation set.
    z = 1.96
    centre = (p + z*z/(2*n))/(1+z*z/n)
    half = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    ax.errorbar(x, p, yerr=[[p-(centre-half)], [(centre+half)-p]], fmt="o",
                color=BLUE, ecolor=BLUE, capsize=3, markersize=4+np.sqrt(n)/3, zorder=3)
    ax.annotate(str(n), (x, centre+half), xytext=(0, 5), textcoords="offset points",
                ha="center", fontsize=8, color=INK)
ax.set(xlim=(.25, 1.02), ylim=(-.02, 1.07))
ax.set_xlabel("区间内平均预测置信度", fontproperties=FONT)
ax.set_ylabel("区间内实际正确率", fontproperties=FONT)
ax.grid(axis='y', color='#E4E7EB', linewidth=.55)
ax.spines[['top','right']].set_visible(False)
save(fig, "图 5.3 置信度可靠性图.pdf", "预测置信度可靠性.pdf")

# Joint scatter with marginal densities; exact paired observations retained.
fig=plt.figure(figsize=(9.2,3.65))
outer=fig.add_gridspec(1,3,left=.07,right=.985,bottom=.18,top=.98,wspace=.38)
axs=[]
for k,(mod,name,color) in enumerate(zip(["text","audio","vision"],MODS,COLORS)):
    grid=outer[k].subgridspec(2,2,height_ratios=[.18,1],width_ratios=[1,.18],hspace=.045,wspace=.045)
    ax=fig.add_subplot(grid[1,0]);top=fig.add_subplot(grid[0,0],sharex=ax);side=fig.add_subplot(grid[1,1],sharey=ax)
    axs.append(ax)
    pairs=[(r['wider_window_transfer'][mod]['random'],r['wider_window_transfer'][mod]['high'])
           for r in rows if r['wider_window_transfer'].get(mod)]
    arr=np.asarray(pairs)
    low=min(0,arr.min());high=max(0,arr.max());margin=max((high-low)*.04,.001)
    limits=(low-margin,high+margin)
    ax.scatter(arr[:,0],arr[:,1],s=7,color=color,alpha=.30,edgecolors='none',rasterized=False)
    ax.plot([low,high],[low,high],ls=(0,(4,3)),color='#606B7C',lw=.8,zorder=0)
    ax.set(xlim=limits,ylim=limits,xlabel=name+'随机窗概率下降')
    ax.set_aspect('equal',adjustable='box')
    ax.text(.045,.96,f"高分窗 > 随机窗：{format_percent4(np.mean(arr[:,1]>arr[:,0]))}\nn = {len(arr)}",
            transform=ax.transAxes,va='top',fontsize=7.5,color='black',linespacing=1.4)
    ax.tick_params(labelsize=8,length=3)
    ax.locator_params(axis='both',nbins=4)
    x=np.linspace(*limits,250)
    dx=gaussian_kde(arr[:,0])(x);dy=gaussian_kde(arr[:,1])(x)
    top.fill_between(x,0,dx,color=color,alpha=.17);top.plot(x,dx,color=color,lw=1)
    side.fill_betweenx(x,0,dy,color=color,alpha=.17);side.plot(dy,x,color=color,lw=1)
    top.set_ylim(0,dx.max()*1.08);side.set_xlim(0,dy.max()*1.08)
    top.axis('off');side.axis('off')
axs[0].set_ylabel('高分窗遮挡后的概率下降')
# Fix marginal positions after enforcing equal units in the scatter panels.
fig.canvas.draw()
for k,ax in enumerate(axs):
    top=fig.axes[3*k+1];side=fig.axes[3*k+2];box=ax.get_position()
    top.set_position([box.x0,box.y1+.009,box.width,.105])
    side.set_position([box.x1+.005,box.y0,.028,box.height])
save(fig,'图 6.4 局部解释忠实度散点.pdf','局部解释忠实度散点.pdf')
