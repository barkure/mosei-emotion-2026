# 本程序及代码是在人工智能工具辅助下完成的
# 人工智能工具名称：DeepSeek，型号：DeepSeek V4.1 Flash，开发公司：杭州深度求索人工智能基础技术研究有限公司，版本颁布日期：2026-09-10

"""Rebuild the existing case figure using frozen attachment-4 explanations.

The x axis is a window ordinal, never a claim of an exact video timestamp.
"""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, fontManager
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'outputs/q3_explanation/q3/attachment4_explanations_full.json'
OUTPUT=ROOT/'paper/figures/解释样本.pdf'
CASES=(('01','text'),('03','text'),('05','vision'))
import sys
sys.path.insert(0, str(ROOT/'src/figures'))
from figure_style import mixed_font, use_serif, black_axis_text
use_serif(ROOT)
CJK=mixed_font()
plt.rcParams.update({'font.size':9})
def main():
    from matplotlib.ticker import MaxNLocator
    from matplotlib.lines import Line2D
    cases={x['sample_id']:x for x in json.loads(SOURCE.read_text(encoding="utf-8"))}
    fig=plt.figure(figsize=(8.8,5.8))
    grid=fig.add_gridspec(3,5,width_ratios=[.62,.20,1.35,.48,2.55],
                          left=.025,right=.975,bottom=.11,top=.94,wspace=0,hspace=.43)
    names=['文本','语音','视频'];colors=['#394D96','#148C84','#CE9A43']
    for row,(ident,modality) in enumerate(CASES):
        sample=cases[ident];contributions=np.asarray(sample['shapley_probability'])
        info=fig.add_subplot(grid[row,0]);info.axis('off')
        strength=sample['strength']
        info_lines = ['样本 '+ident,
                      '预测'+['负向','中性','正向'][sample['polarity']],
                      '强度 '+f'{strength:.4f}']
        for line_index, line in enumerate(info_lines):
            info.text(.04, .72 - .25 * line_index, line, transform=info.transAxes,
                      fontproperties=CJK, fontsize=9, color='black', va='baseline')
        left=fig.add_subplot(grid[row,2]);right=fig.add_subplot(grid[row,4])
        left.axvline(0,color='#8A929C',lw=.8,ls=(0,(3,3)))
        for j,(val,color) in enumerate(zip(contributions,colors)):
            y=2-j
            left.hlines(y,0,val,color=color,lw=1.4)
            left.scatter([val],[y],s=40,color=color,edgecolor='white',linewidth=.7,zorder=3)
            beside = ident == '05' and j == 1
            left.annotate(f'{val:+.4f}',(val,y),xytext=(-9,0) if beside else (0,9),
                          textcoords='offset points',ha='right' if beside else 'center',
                          va='center' if beside else 'bottom',fontsize=8)
        left.set_yticks([2,1,0],names,fontproperties=CJK)
        left.set_xlim(-.16,.33);left.set_ylim(-.4,2.65)
        left.set_xticks([-.1,0,.1,.2,.3])
        left.spines['left'].set_visible(False);left.tick_params(axis='y',length=0,pad=4)
        if row==2:left.set_xlabel('Shapley 概率贡献',fontproperties=CJK)
        scores=np.asarray(sample['local'][modality]['all_window_drops']);xs=np.arange(1,len(scores)+1)
        color=colors[['text','audio','vision'].index(modality)]
        high=int(np.argmax(scores));span=max(float(np.ptp(scores)),.015)
        right.axhline(0,color='#8A929C',lw=.8,ls=(0,(3,3)))
        right.fill_between(xs,0,scores,color=color,alpha=.09,lw=0)
        right.plot(xs,scores,color=color,lw=1.4)
        right.axvline(high+1,color=color,lw=.7,ls=(0,(2,3)),alpha=.5)
        right.scatter([high+1],[scores[high]],s=37,marker='D',facecolor='white',edgecolor=color,linewidth=1.3,zorder=4)
        right.annotate(f'{scores[high]:.4f}',(high+1,scores[high]),xytext=(0,9),textcoords='offset points',ha='center',va='bottom',fontsize=8)
        right.set_xlim(.6,len(scores)+.4);right.set_ylim(min(0,float(scores.min()))-.13*span,float(scores.max())+.22*span)
        right.xaxis.set_major_locator(MaxNLocator(nbins=5,integer=True))
        right.yaxis.set_major_locator(MaxNLocator(nbins=4,steps=[1,2,5,10],min_n_ticks=4))
        right.grid(axis='y',color='#E8ECF0',lw=.5);right.set_axisbelow(True)
        right.set_ylabel('预测类别概率下降',fontproperties=CJK)
        if row==2:right.set_xlabel('候选窗口序号',fontproperties=CJK)
        for ax in (left,right):
            ax.spines[['top','right']].set_visible(False)
            ax.tick_params(axis='x',length=3,labelsize=8,color='black')
            ax.tick_params(axis='y',labelsize=8,colors='black')
    selected=Line2D([],[],marker='D',mfc='white',mec='#626874',ls='none',label='所选窗口')
    fig.legend(handles=[selected],loc='upper right',bbox_to_anchor=(.98,1),frameon=False,prop=CJK)
    OUTPUT.parent.mkdir(parents=True,exist_ok=True)
    black_axis_text(fig)
    fig.savefig(OUTPUT,bbox_inches='tight',pad_inches=.1);plt.close(fig)
    print(OUTPUT.relative_to(ROOT))
if __name__=='__main__':main()
