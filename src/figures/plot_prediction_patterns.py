# 本程序及代码是在人工智能工具辅助下完成的
# 人工智能工具名称：DeepSeek，型号：DeepSeek V4.1 Flash，开发公司：杭州深度求索人工智能基础技术研究有限公司，版本颁布日期：2026-09-10

from figure_style import mixed_font
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.path import Path as MPath
from matplotlib.patches import PathPatch,Rectangle
from matplotlib.lines import Line2D
from matplotlib.font_manager import FontProperties
from figure_style import use_serif,prepare,BLUE,TEAL,GOLD,PALE
ROOT=Path(__file__).resolve().parents[2]
FONT=mixed_font()
use_serif(ROOT)
plt.rcParams.update({'font.size':9})
OUT=ROOT/'paper/figures'
d=json.loads((ROOT/'outputs/q2_prediction/q2_systematic/gate_recon_ensemble.json').read_text(encoding="utf-8"))
rows=json.loads((ROOT/'outputs/q3_explanation/q3/validation_explanation_all.json').read_text(encoding="utf-8"))['details']
assert len(rows)==728 and len({r['valid_index'] for r in rows})==728
cm=np.zeros((3,3),int)
for r in rows:cm[r['true_polarity'],r['polarity']]+=1
assert np.trace(cm)/cm.sum()==d['complete']['accuracy']
print('confusion:',cm.tolist())
colors=[BLUE,TEAL,GOLD];names=['负向','中性','正向']
fig,ax=plt.subplots(figsize=(7.6,3.6));fig.subplots_adjust(left=.03,right=.97,bottom=.12,top=.99)
# Ribbon widths use the same sample-count scale on both sides.
gap=65;left=np.r_[0,np.cumsum(cm.sum(1)+gap)[:-1]];right=np.r_[0,np.cumsum(cm.sum(0)+gap)[:-1]]
lcursor=left.astype(float).copy();rcursor=right.astype(float).copy()
for i in range(3):
 for j in range(3):
  n=cm[i,j];a=lcursor[i];b=rcursor[j]
  verts=[(.12,a),(.44,a),(.56,b),(.88,b),(.88,b+n),(.56,b+n),(.44,a+n),(.12,a+n),(.12,a)]
  codes=[MPath.MOVETO,*([MPath.CURVE4]*3),MPath.LINETO,*([MPath.CURVE4]*3),MPath.CLOSEPOLY]
  ax.add_patch(PathPatch(MPath(verts,codes),facecolor=colors[i],alpha=.60 if i==j else .16,edgecolor='white',lw=.6))
  ax.text(.145,a+n/2,str(n),ha='left',va='center',fontsize=8,color='black')
  lcursor[i]+=n;rcursor[j]+=n
for i in range(3):
 for x,ys,totals,align,tx in [(.10,left,cm.sum(1),'right',.075),(.88,right,cm.sum(0),'left',.925)]:
  ax.add_patch(Rectangle((x,ys[i]),.02,totals[i],facecolor=colors[i],edgecolor='white',lw=.7))
  ax.text(tx,ys[i]+totals[i]/2,f'{names[i]}\n{totals[i]}条',ha=align,va='center',fontproperties=FONT,linespacing=1.5)
ax.set_xlim(-.04,1.04);ax.set_ylim(870,-15);ax.axis('off')
ax.text(.11,-.04,'真实极性',transform=ax.transAxes,fontproperties=FONT,ha='center')
ax.text(.89,-.04,'预测极性',transform=ax.transAxes,fontproperties=FONT,ha='center')
prepare(fig);fig.savefig(OUT/'极性预测流向.pdf',bbox_inches='tight',pad_inches=.1);plt.close(fig)
# Six scenarios in the classification/regression plane; arrows go from 20% to 40%.
fig,axes=plt.subplots(1,2,figsize=(8,3.25));fig.subplots_adjust(left=.095,right=.98,bottom=.22,top=.84,wspace=.34)
positions=['start','middle','end'];pn=['首部','中部','尾部']
for pos,name,c in zip(positions,pn,colors):
 rr=[next(r for r in d['scenarios'] if r['modality']=='text' and r['fraction']==f and r['position']==pos) for f in [.2,.4]]
 x=[r['macro_f1'] for r in rr];y=[r['mae_consistent'] for r in rr]
 axes[0].annotate('',xy=(x[1],y[1]),xytext=(x[0],y[0]),arrowprops=dict(arrowstyle='->',color=c,lw=1.3,shrinkA=6,shrinkB=6))
 for k in range(2):axes[0].plot(x[k],y[k],marker='o' if k==0 else 's',ms=6,color=c,mec='white',mew=.7)
 axes[0].annotate(name,(x[0],y[0]),xytext=(-8,-9) if pos=='middle' else (7,0),textcoords='offset points',ha='right' if pos=='middle' else 'left',va='top' if pos=='middle' else 'center',fontproperties=FONT,fontsize=8)
 z=[r['per_class_f1'][1] for r in rr]
 axes[1].plot([0,1],z,color=c,lw=1.3,label=name)
 for k in range(2):
  axes[1].plot(k,z[k],marker='o' if k==0 else 's',ms=6,color=c,mec='white',mew=.7)
  axes[1].annotate(f'{z[k]:.4f}',(k,z[k]),xytext=(-8 if k==0 else 8,0),textcoords='offset points',ha='right' if k==0 else 'left',va='center',fontsize=8)
axes[0].plot(d['complete']['macro_f1'],d['complete']['mae_consistent'],marker='D',color='#626874',ms=5)
axes[0].annotate('完整输入',(d['complete']['macro_f1'],d['complete']['mae_consistent']),xytext=(-7,4),textcoords='offset points',ha='right',fontproperties=FONT,fontsize=8)
axes[0].set(xlim=(.543,.635),ylim=(.57,.645),xlabel='Macro-F1');axes[0].set_ylabel('投影后 MAE',fontproperties=FONT)
axes[1].axhline(d['complete']['per_class_f1'][1],color='#9097A0',lw=.8,ls=(0,(3,3)))
axes[1].set_xlim(-.45,1.45);axes[1].set_ylim(.39,.49)
axes[1].set_xticks([0,1],['20%','40%']);axes[1].set_xlabel('文本缺失比例',fontproperties=FONT);axes[1].set_ylabel('中性类别 F1',fontproperties=FONT)
for ax in axes:
 ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',color=PALE,lw=.6);ax.set_axisbelow(True)
handles=[Line2D([],[],color=c,lw=2,label=n) for c,n in zip(colors,pn)]
handles += [Line2D([],[],color='#626874',marker=m,ls='none',label=n) for m,n in [('o','缺失20%'),('s','缺失40%')]]
fig.legend(handles=handles,loc='upper center',ncol=5,frameon=False,prop=FONT,columnspacing=1.7)
prepare(fig);fig.savefig(OUT/'文本缺失位置效应.pdf',bbox_inches='tight',pad_inches=.1);plt.close(fig)
