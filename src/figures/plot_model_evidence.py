# 本程序及代码是在人工智能工具辅助下完成的
# 人工智能工具名称：ChatGPT，型号：GPT-6 Astra，开发公司：OpenAI，版本颁布日期：2026-09-04

from figure_style import mixed_font
"""Chapter 5 model structure and measured component/input comparisons."""
from pathlib import Path
import json
import sys
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib import patheffects
from matplotlib.font_manager import FontProperties
from figure_style import BLUE, TEAL, GOLD, PALE, prepare, use_serif
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'paper/figures'
DATA=ROOT/'outputs/q2_prediction/q2_systematic'
use_serif(ROOT)
FONT=mixed_font()
plt.rcParams.update({'font.size':9})
def save(fig,name):
    if getattr(fig, '_module_text_layouts', None):
        fig.canvas.draw()
        renderer=fig.canvas.get_renderer()
        for axis,x,y,w,h,artists,size in fig._module_text_layouts:
            bounds=[matplotlib.transforms.Bbox.union([a.get_window_extent(renderer) for a in line]) for line in artists]
            bottom=axis.transData.transform((x,y))[1]
            top=axis.transData.transform((x,y+h))[1]
            gap=min(size*fig.dpi/72*.35,
                    max(0,((top-bottom)*.80-sum(b.height for b in bounds))/max(1,len(bounds)-1)))
            block_height=sum(b.height for b in bounds)+gap*(len(bounds)-1)
            cursor=(top+bottom+block_height)/2
            for line,bound in zip(artists,bounds):
                widths=[a.get_window_extent(renderer).width for a in line]
                left=axis.transData.transform((x+w/2,y))[0]-sum(widths)/2
                for artist,width in zip(line,widths):
                    _,py=axis.transData.transform(artist.get_position())
                    artist.set_position(axis.transData.inverted().transform((left,py+cursor-bound.y1)))
                    left+=width
                cursor-=bound.height+gap
    prepare(fig)
    fig.savefig(OUT/name,bbox_inches='tight',pad_inches=.10)
    plt.close(fig)
    print(name)
def read(name):
    d=json.loads((DATA/(name+'.json')).read_text(encoding="utf-8"))
    assert d['data_version']=='aligned_50' and d['valid_n']==728
    return d

# Explicit ports keep every arrow attached to its source and destination.
fig,ax=plt.subplots(figsize=(10.0,5.85))
ax.set(xlim=(-.1,14.0),ylim=(-.15,7.85));ax.axis('off')
nodes={}
def box(key,x,y,w,h,text,color=BLUE,fill='white',size=9,dashed=False,heading=True):
    nodes[key]=(x,y,w,h)
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=0.055',
        lw=.9,edgecolor=color,facecolor=fill,linestyle=(0,(4,3)) if dashed else '-',zorder=3))
    artists=[]
    for i,line in enumerate(text.split('\n')):
        # Separate formulas from prose: mathtext cannot fall back to Chinese fonts.
        parts=[]
        for part in filter(None,re.split(r'(\$[^$]+\$)',line)):
            artist=ax.text(x+w/2,y+h/2,part,ha='left',va='baseline',fontproperties=FONT,
                           fontsize=size,color='black',zorder=4)
            if heading and i==0:
                artist.set_path_effects([patheffects.withStroke(linewidth=.25,foreground='black')])
            parts.append(artist)
        artists.append(parts)
    if not hasattr(fig, '_module_text_layouts'):
        fig._module_text_layouts=[]
    fig._module_text_layouts.append((ax,x,y,w,h,artists,size))
def port(key,side,offset=0):
    x,y,w,h=nodes[key]
    return {'left':(x,y+h/2+offset),'right':(x+w,y+h/2+offset),
            'top':(x+w/2+offset,y+h),'bottom':(x+w/2+offset,y)}[side]
def link(source,target,color='#6E7B88',out='right',into='left',via=(),src_offset=0,dst_offset=0,dashed=False):
    points=[port(source,out,src_offset),*via,port(target,into,dst_offset)]
    for a,b in zip(points[:-2],points[1:-1]):
        ax.plot([a[0],b[0]],[a[1],b[1]],color=color,lw=.95,ls='--' if dashed else '-',zorder=1)
    ax.add_patch(FancyArrowPatch(points[-2],points[-1],arrowstyle='-|>',mutation_scale=10,
        shrinkA=0,shrinkB=0,lw=.95,color=color,linestyle='--' if dashed else '-',zorder=2))

box('text',.1,6.2,3.6,1.4,'文本词元与掩码\n冻结 BERT，结构掩码控制注意力\nCLS + 已观测词元均值：1536维',BLUE,'#F1F3FA',9,heading=True)
box('text_mlp',4.7,6.3,2.6,1.2,'文本专家 MLP\n1536 → 192 → 128\n隐表示 $z^x$',BLUE,size=9)
box('text_heads',8.1,6.2,1.8,1.4,'专家预测头\n分类：3维\n回归：1维\n$o^x$，$y^x$',BLUE,size=9)
link('text','text_mlp',BLUE);link('text_mlp','text_heads',BLUE)
left_gap = (6.2 - 1.4 - 1.44 - 2 * 1.36) / 3
audio_y = 6.2 - left_gap - 1.36
video_y = audio_y - left_gap - 1.36
box('audio',.1,audio_y,3.6,1.36,'音频序列与观测掩码\n50 × 74 → 64维残差 Conv1D\n掩码均值 + 掩码注意力池化\n拼接观测比例：129维',TEAL,'#F0F7F6',8.8,heading=True)
box('vision',.1,video_y,3.6,1.36,'视频序列与观测掩码\n50 × 35 → 64维残差 Conv1D\n掩码均值 + 掩码注意力池化\n拼接观测比例：129维',GOLD,'#FBF7EE',8.8,heading=True)
box('av_mlp',4.7,3.9,2.6,1.2,'音视频专家 MLP\n258 → 192 → 128\n隐表示 $r^{av}$',TEAL,size=9)
box('av_heads',8.1,3.8,1.8,1.4,'专家预测头\n分类：3维\n回归：1维\n$o^{av}$，$y^{av}$',TEAL,size=9)
link('audio','av_mlp',TEAL,via=[(4.15,port('audio','right')[1]),(4.15,4.75)],dst_offset=.25)
link('vision','av_mlp',GOLD,via=[(4.15,port('vision','right')[1]),(4.15,4.25)],dst_offset=-.25)
link('av_mlp','av_heads',TEAL)
box('fusion',10.8,5.3,2.95,1.3,'MoE 输出融合\n$o=go^x+(1-g)o^{av}$\n$y=gy^x+(1-g)y^{av}$',BLUE,'#F1F3FA',9)
link('text_heads','fusion',BLUE,via=[(10.3,6.9),(10.3,6.3)],dst_offset=.35)
link('av_heads','fusion',TEAL,via=[(10.05,4.5),(10.05,5.95)])
box('ensemble',10.8,2.92,2.95,1.48,'三种子 MoE 集成\n各自训练，输出分别取均值\n平均 logits → 极性类别\n平均原始强度 → 符号投影',BLUE,size=8.8)
link('fusion','ensemble',BLUE,out='bottom',into='top')

# Place the training head directly below the expert supplying its representation.
box('rec_map',4.7,1.62,2.6,1.0,'线性重构头\n128 → 1536',TEAL,size=9,dashed=True)
box('mse',.1,1.40,3.6,1.44,'MSE 损失，权重0.02\n仅在文本未遮挡样本上计算\n目标：1536维 BERT 表示\n目标停止梯度',TEAL,'#F0F7F6',8.8,True)
link('av_mlp','rec_map',TEAL,out='bottom',into='top',dashed=True)
link('rec_map','mse',TEAL,out='left',into='right',dashed=True)
ax.text(6.12,3.3,'仅训练',ha='left',va='center',fontproperties=FONT,
        fontsize=8.5,color='black',zorder=5)

ax.text(12.275, 2.30, '音视频编码细节\n位置编码 + 核宽3卷积\n卷积前后施加观测掩码',
        ha='center', va='top', fontproperties=FONT, fontsize=8.5, linespacing=1.5)

# Gate computation uses the named expert representations and observed fractions.
box('gate_inputs',.1,.02,4.6,.95,'门控输入拼接\n文本 $z^x$ + 音视频 $r^{av}$ + 三模态观测比例',GOLD,'#FBF7EE',9)
box('gate',5.3,.02,2.9,.95,'门控 MLP + Sigmoid\n259 → 64 → 1',GOLD,size=9)
box('boundary',8.8,.02,1.2,.95,'可用性\n约束 $g$',GOLD,size=9)
link('gate_inputs','gate',GOLD);link('gate','boundary',GOLD)
link('boundary','fusion',GOLD,via=[(10.5,.495),(10.5,5.55)],dst_offset=-.4)
save(fig,'混合专家模型框架.pdf')
if '--architecture-only' in sys.argv:
    sys.exit(0)

# Transposed overview with a curved connection to the magnified range.
from matplotlib.patches import ConnectionPatch
d=read('gate_recon_ensemble')
baseline=d['complete']['macro_f1']
fig,axes=plt.subplots(1,2,figsize=(8.4,3.65),
                       gridspec_kw={'width_ratios':[1.12,1]})
fig.subplots_adjust(left=.095,right=.965,bottom=.20,top=.84,wspace=.33)
colors=[BLUE,TEAL,GOLD];names=['文本','语音','视频']
zoom_lo,zoom_hi=-.023,.021
zoom_color='#E7EDF3'
for ax in axes:
    ax.axhline(0,color='#737C87',lw=.8,ls=(0,(3,3)),zorder=1)
    ax.set_xticks([0,1,2],['20%','40%','全部'],fontproperties=FONT)
    ax.set_xlim(-.48,2.48)
    for boundary in (.5,1.5):
        ax.axvline(boundary,color='#CBD3DD',lw=.6,zorder=1)
    ax.spines[['top','right']].set_visible(False)
    ax.tick_params(axis='both',length=3,color='black')
    ax.set_xlabel('移除比例',fontproperties=FONT)
axes[0].axhspan(zoom_lo,zoom_hi,color=zoom_color,zorder=0)
axes[1].set_facecolor(zoom_color)
for mi,(mod,name,c) in enumerate(zip(['text','audio','vision'],names,colors)):
    for ri,frac in enumerate([.2,.4,1.]):
        vals=np.array([r['macro_f1']-baseline for r in d['scenarios']
                       if r['modality']==mod and r['fraction']==frac])
        assert len(vals)==(1 if frac==1. else 3)
        avg=float(vals.mean());x=ri+(mi-1)*.22
        for ai,ax in enumerate(axes):
            if ai==1 and mod=='text':continue
            ax.errorbar(x,avg,yerr=[[avg-vals.min()],[vals.max()-avg]],
                        fmt='o',color=c,mec='white',mew=.7,ms=6,
                        elinewidth=1.4,capsize=3,zorder=3,
                        label=name if ri==0 and ai==0 else None)
            if ai==1 or mod=='text':
                label=f'{avg if abs(avg) >= 0.00005 else 0.0:+.4f}'
                above=ai==1 and mod=='vision' and ri<2
                anchor=vals.max() if above else vals.min()
                ax.annotate(label,(x,anchor),xytext=(0,9 if above else -10),
                            textcoords='offset points',ha='center',
                            va='bottom' if above else 'top',fontsize=7.5,color='black')
axes[0].set_ylim(-.245,.033)
axes[0].set_yticks([-.2,-.15,-.1,-.05,0],['−0.20','−0.15','−0.10','−0.05','0'])
axes[1].set_ylim(zoom_lo,zoom_hi)
axes[1].set_yticks([-.02,-.01,0,.01,.02],['−0.02','−0.01','0','+0.01','+0.02'])
axes[0].set_ylabel('Macro-F1 变化',fontproperties=FONT)
# Place detail ticks on its outer edge to keep the curved expansion unobstructed.
axes[1].yaxis.tick_right()
axes[1].spines['left'].set_visible(False)
axes[1].spines['right'].set_visible(True)
for edge in [zoom_hi,zoom_lo]:
    curve=ConnectionPatch(xyA=(2.48,edge),xyB=(-.48,edge),
                          coordsA='data',coordsB='data',axesA=axes[0],axesB=axes[1],
                          arrowstyle='-',connectionstyle='arc3,rad=0',
                          lw=.9,color='#99A8B8',clip_on=False,zorder=0)
    fig.add_artist(curve)
handles,labels=axes[0].get_legend_handles_labels()
fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.99),
           ncol=3,frameon=False,prop=FONT,handletextpad=.6,columnspacing=2.5)
save(fig,'模态移除敏感性.pdf')
