# 本程序及代码是在人工智能工具辅助下完成的
# 人工智能工具名称：ChatGPT，型号：GPT-6 Astra，开发公司：OpenAI，版本颁布日期：2026-09-04

"""Print-sized Chapter 4 figures from attachment 1 audit outputs.

Run from the repository root with ``pixi run python src/figures/plot_problem1_overview.py``.
"""
from __future__ import annotations
from figure_style import mixed_font, discrete_heatmap

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
import pandas as pd

from figure_style import BLUE, TEAL, GOLD, RED, INK, PALE, NEUTRAL, sequential_cmap, use_serif, black_axis_text

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper/figures"
CJK = mixed_font()
use_serif(ROOT)
plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#000000"})


def style(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=PALE, lw=0.6)
    ax.tick_params(length=2, labelsize=8, colors="#000000")
    for spine in ax.spines.values():
        spine.set_color("#000000")


def save(fig, name, gallery_name=None):
    black_axis_text(fig)
    fig.savefig(OUT / name, bbox_inches="tight", pad_inches=0.09)
    if gallery_name is not None:
        gallery_target = ROOT / "outputs/figures" / gallery_name
        fig.savefig(gallery_target, bbox_inches="tight", pad_inches=0.09)
        print(gallery_target)
    plt.close(fig)
    print(OUT / name)


def timebase_figure():
    data = pd.read_csv(ROOT / "outputs/q1_features/attachment1_timebase/summary.csv")
    assert len(data) == 100 and data.stable_id.nunique() == 100
    delta_ms = -data.av_end_coverage_offset_s.to_numpy() * 1000
    assert int(np.count_nonzero(delta_ms > 0)) == 98
    assert int(np.count_nonzero(np.abs(delta_ms) > 23.2)) == 96
    ordered = np.sort(delta_ms)
    rank = np.arange(1, ordered.size + 1)
    median = float(np.median(ordered))

    fig, ax = plt.subplots(figsize=(7.2, 2.55))
    fig.subplots_adjust(left=0.115, right=0.985, bottom=0.24, top=0.95)
    ax.axhspan(-23.2, 23.2, color=PALE, alpha=0.7, zorder=0)
    ax.axhline(0, color=NEUTRAL, lw=0.85, zorder=1)
    ax.axhline(median, color="#666666", lw=0.85, ls=(0, (3, 3)), zorder=1)
    ax.scatter(rank[ordered >= 0], ordered[ordered >= 0], s=19, color=BLUE,
               edgecolor="white", linewidth=0.35, zorder=3)
    ax.scatter(rank[ordered < 0], ordered[ordered < 0], s=25, color=GOLD,
               edgecolor="white", linewidth=0.35, zorder=4)
    ax.text(2, 184, "96/100 超过 AAC 帧长", fontsize=8.5,
            color=INK, ha="left", va="top")
    ax.text(99, median + 5, f"中位数 {median:.4g} ms",
            fontsize=8.5, color="#000000", ha="right", va="bottom")
    ax.text(99, 16, "AAC ±23.2 ms", fontsize=8.5, color="#000000", ha="right", va="center")
    ax.set_xlim(0, 101)
    ax.set_ylim(-55, 195)
    ax.set_xticks([1, 25, 50, 75, 100])
    ax.set_yticks([-50, 0, 50, 100, 150])
    ax.set_xlabel("按终点差从小到大排序的样本", fontproperties=CJK)
    ax.set_ylabel("视频终点－音频终点 / ms", fontproperties=CJK)
    style(ax)
    save(fig, "音画时间基准.pdf", None)



def coverage_figure():
    data = pd.read_csv(ROOT / 'outputs/q1_features/q1_final/word_alignment_summary.csv')
    categories = [data.accepted_words.eq(data.official_words),
                  data.accepted_words.gt(0) & data.accepted_words.lt(data.official_words),
                  data.category.eq('content_mismatch'), data.category.eq('other_language'),
                  data.category.eq('no_audible_speech')]
    labels = ['全部词时间可用', '部分词时间可用', '音文不一致', '其他语言', '无对应语音']
    colors = [BLUE, TEAL, '#A7AFBA', '#C8C0B3', '#DCE0E5']
    fig, ax = plt.subplots(figsize=(7.6, 2.8))
    positions=[0,1,2.5,3.5,4.5]
    aligned=int(data.accepted_words.gt(0).sum())
    unavailable=len(data)-aligned
    assert sum(int(mask.sum()) for mask in categories)==len(data)
    for y, mask, color in zip(positions, categories, colors):
        n = int(mask.sum())
        ax.barh(y, n, color=color, height=.52)
        ax.text(n+1, y, str(n), va='center', fontsize=9)
    for low, high, total, label in [(-.26,1.26,aligned,'词级对齐'),
                                     (2.24,4.76,unavailable,'原生特征保留')]:
        ax.plot([53,54,54,53],[low,low,high,high],color='#8B929D',
                lw=.8,clip_on=False)
        ax.text(56,(low+high)/2,f'{total} 条\n{label}',ha='left',va='center',
                fontsize=8.5,color='black',linespacing=1.5,clip_on=False)
    ax.set_yticks(positions, labels);ax.set_ylim(5.05,-.65);ax.set_xlim(0,52)
    ax.set_xticks([0,10,20,30,40,50])
    ax.set_xlabel('样本数');style(ax); ax.grid(False)
    ax.spines['left'].set_visible(True);ax.tick_params(axis='y',length=2,pad=7)
    fig.subplots_adjust(left=.18,right=.84,bottom=.22,top=.97)
    save(fig,'音文筛查分布.pdf')


def selection_figure():
    import json
    from matplotlib import patheffects
    from collections import Counter
    from matplotlib.patches import Rectangle
    reasons=Counter()
    for path in (ROOT/'outputs/q1_features/attachment1_alignment/recovered').glob('*.json'):
        record=json.loads(path.read_text(encoding="utf-8"))
        for word in record.get('words',[]):
            reasons[word['reason']]+=1
    usable=reasons['accepted_automatic_with_content_evidence']
    exception=sum(reasons[k] for k in ['content_mismatch','other_language','no_audible_speech'])
    local=sum(reasons.values())-exception-usable
    assert (sum(reasons.values()),exception,local,usable)==(1932,296,103,1533)
    fig,ax=plt.subplots(figsize=(8.0,3.5))
    ax.set_xlim(-.1,9.7);ax.set_ylim(-.61,3.6);ax.axis('off')
    def node(x,y,w,h,heading,value,color):
        ax.add_patch(Rectangle((x,y),w,h,facecolor='white',edgecolor='#C7CED8',lw=.75))
        ax.plot([x,x+w],[y+h,y+h],color=color,lw=2,solid_capstyle='butt')
        ax.text(x+w/2,y+h*.71,heading,ha='center',va='center',fontsize=10.5,color='black',
                path_effects=[patheffects.withStroke(linewidth=.22, foreground='black')])
        ax.text(x+w/2,y+h*.30,value,ha='center',va='center',fontsize=9.5,color=color)
    def arrow(a,b):
        ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',lw=.9,color='#737D8C',shrinkA=3,shrinkB=3))
    node(0,2.52,2.6,.90,'100 条原始样本',f'{sum(reasons.values())} 个词位置',INK)
    node(3.35,2.52,2.6,.90,'80 条词级对齐样本',f'{usable+local} 个词位置',BLUE)
    node(6.70,2.52,2.6,.90,'词时间可用',f'{usable} 个词位置',TEAL)
    arrow((2.6,2.97),(3.35,2.97));arrow((5.95,2.97),(6.7,2.97))
    def detail_box(x, heading, subtitle, color, rows, footer=()):
        bottom, height, width = -.55, 2.43, 2.6
        ax.add_patch(Rectangle((x, bottom), width, height,
                               facecolor='white', edgecolor='#C7CED8', lw=.75))
        ax.plot([x, x+width], [bottom+height]*2, color=color, lw=2,
                solid_capstyle='butt')
        ax.text(x+width/2, 1.62, heading, ha='center', va='center', fontsize=10.5,
                path_effects=[patheffects.withStroke(linewidth=.22, foreground='black')])
        ax.text(x+width/2, 1.28, subtitle, ha='center', va='center', fontsize=9.5, color=color)
        inset = .32
        ax.plot([x+inset, x+width-inset], [1.04]*2, color='#E0E4EA', lw=.6)
        # Equal-height flex cells within the detail area; headings stay fixed.
        top, low = 1.04, (.10 if footer else bottom+.08)
        cell_height = (top-low)/len(rows)
        for i, (label, value) in enumerate(rows):
            y = top-(i+.5)*cell_height
            ax.text(x+inset, y, label, ha='left', va='center', fontsize=8.5)
            ax.text(x+width-inset, y, value, ha='right', va='center', fontsize=8.5)
        if footer:
            ax.text(x+width/2, -.12, '\n'.join(footer), ha='center', va='center',
                    fontsize=8.5, linespacing=1.5)
    detail_box(0, '20 条异常样本', f'{exception} 个词位置', '#747C87',
               [(label, str(reasons[key])) for label, key in [
                   ('音文不一致', 'content_mismatch'), ('其他语言', 'other_language'),
                   ('无对应语音', 'no_audible_speech')]])
    detail_box(3.35, '局部词时间不可用', f'{local} 个词位置', '#747C87',
               [(label, str(reasons[key])) for label, key in [
                   ('低置信', 'low_character_confidence'),
                   ('短语范围外', 'outside_human_confirmed_phrase'),
                   ('静音区间', 'inside_detected_silence'),
                   ('无目标字符', 'no_supported_target_characters'),
                   ('时间跨度异常', 'excessive_word_span')]])
    detail_box(6.7, '按词时间汇聚音视频', '生成词级三模态特征', TEAL,
               [('文本', '768 维'), ('音频', '17 维'), ('视频', '52 维')],
               footer=('三模态第 i 行对应官方词 i，', '缺失特征用掩码标记。'))
    for x in (1.3, 4.65, 8):
        arrow((x,2.52),(x,1.88))
    fig.subplots_adjust(left=.02,right=.99,bottom=.04,top=.98)
    save(fig,'词时间筛选.pdf')


def observation_figure():
    d=pd.read_csv(ROOT/'outputs/q1_features/q1_final/word_alignment_summary.csv')
    time_rate=(d.accepted_words/d.official_words).to_numpy()
    face_rate=(d.face_words/d.official_words).to_numpy()
    assert len(d)==100 and np.all(face_rate<=time_rate)
    # Exact empirical CDF: F(x) = number of samples with coverage <= x / 100.
    x=np.unique(np.r_[0., time_rate, face_rate, 1.])
    ft=(time_rate[:,None]<=x).mean(axis=0)
    ff=(face_rate[:,None]<=x).mean(axis=0)
    fig,ax=plt.subplots(figsize=(6.5,3.45))
    ax.fill_between(x,ft,ff,step='post',color='#E9EEF3',zorder=1)
    ax.step(x,ft,where='post',color=BLUE,lw=1.6,label='词时间可用',zorder=3)
    ax.step(x,ff,where='post',color=TEAL,lw=1.6,label='面部特征可用',zorder=3)
    ax.set(xlim=(-.015,1.015),ylim=(0,1.035),xticks=[0,.2,.4,.6,.8,1],yticks=[0,.2,.4,.6,.8,1])
    ax.set_xlabel('覆盖率');ax.set_ylabel('累计样本比例')
    ax.legend(loc='upper left',frameon=False,fontsize=9)
    style(ax)
    fig.subplots_adjust(left=.12,right=.975,bottom=.19,top=.97)
    save(fig,'模态来源概览.pdf')


def typical_figure():
    import subprocess,io
    from PIL import Image
    from matplotlib.patches import Rectangle
    d=dict(np.load(ROOT/'outputs/q1_features/q1_final/features/-egA8-b7-3M__9.npz'))
    start=d['word_start_s'];end=d['word_end_s'];words=d['official_words']
    fig=plt.figure(figsize=(9,6.25))
    # Shared left and right margins for every temporal panel.
    gs=fig.add_gridspec(5,1,left=.10,right=.985,bottom=.10,top=.97,hspace=.18,height_ratios=[1.45,.60,1,1,1.15])
    top=fig.add_subplot(gs[0]);top.axis('off')
    video=ROOT/str(d['relative_path'].item())
    for pos,t in zip([.05,.285,.52,.755],[.80,1.78,3.05,4.25]):
        frame=subprocess.check_output(['ffmpeg','-loglevel','error','-ss',str(t),'-i',str(video),'-frames:v','1','-vf',r'crop=min(iw\,ih):min(iw\,ih),scale=220:220','-f','image2pipe','-vcodec','png','-'])
        ax=top.inset_axes([pos,0,.19,.96]);ax.imshow(Image.open(io.BytesIO(frame)));ax.axis('off')
        ax.text(.035,.965,f'{t:.2f} s',ha='left',va='top',
                transform=ax.transAxes,fontsize=8,color='black',
                bbox=dict(facecolor='white',edgecolor='none',boxstyle='square,pad=0.22'),
                zorder=5)
    axes=[fig.add_subplot(gs[i]) for i in range(1,5)]
    for ax in axes:
        ax.set_xlim(0,float(d['video_clip_range_s'][1]));ax.tick_params(labelsize=8)
        for a,b in zip(start,end): ax.axvspan(a,b,color=BLUE,alpha=.035,lw=0)
    ax=axes[0];ax.set_ylim(0,1);ax.set_yticks([]);ax.set_ylabel('')
    for i,(a,b,w) in enumerate(zip(start,end,words)):
        ax.add_patch(Rectangle((a,.18),b-a,.18,facecolor=BLUE,alpha=.65,lw=0))
        ax.text((a+b)/2,.58 if i%2==0 else .80,w.strip('.,'),fontsize=7.7,ha='center',va='center')
    ax.spines['left'].set_visible(False)
    times=(d['audio_window_media_start_s']+d['audio_window_media_end_s'])/2
    axes[1].plot(times,d['audio_window_features'][:,13],color=TEAL,lw=.9)
    axes[1].set_ylabel('音频对数 RMS');style(axes[1])
    names=d['blendshape_names'].tolist();feature=d['vision_frame_features'][:,names.index('jawOpen')].copy()
    feature[~d['vision_frame_single_face_valid']]=np.nan
    axes[2].plot(d['vision_frame_display_times_s'],feature,color=BLUE,lw=1)
    axes[2].set_ylabel('视频张口系数');style(axes[2])
    ai=np.diff(d['word_audio_offsets']);vi=np.diff(d['word_video_offsets']);mid=(start+end)/2
    axes[3].plot(mid,ai,'o-',color=TEAL,lw=.9,ms=3,label='语音窗')
    axes[3].plot(mid,vi,'s-',color=BLUE,lw=.9,ms=3,label='视频帧')
    axes[3].set_ylabel('来源数量');axes[3].set_xlabel('时间 / s');style(axes[3])
    axes[3].legend(frameon=False,ncol=1,loc='upper right',fontsize=8, labelspacing=.6)
    for ax in axes[:-1]:ax.tick_params(axis='x',labelbottom=False)
    for ax in axes:ax.yaxis.set_label_coords(-.045,.5)
    save(fig,'典型样本对齐.pdf')


if __name__=='__main__':
    timebase_figure();coverage_figure();selection_figure();observation_figure();typical_figure()
