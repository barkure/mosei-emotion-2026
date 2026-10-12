# 本程序及代码是在人工智能工具辅助下完成的
# 人工智能工具名称：DeepSeek，型号：DeepSeek V4.1 Flash，开发公司：杭州深度求索人工智能基础技术研究有限公司，版本颁布日期：2026-09-10

"""Validation-only strength diagnostics from the frozen ensemble explanations.

Included in Chapter 5; sync_paper_gallery.py assigns its paper number.
"""
from pathlib import Path
import json
import csv
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from figure_style import use_serif, prepare, BLUE, TEAL, GOLD, PALE

ROOT = Path(__file__).resolve().parents[2]
use_serif(ROOT)
plt.rcParams.update({'font.size': 9})
rows = sorted(json.loads((ROOT/'outputs/q3_explanation/q3/validation_explanation_all.json').read_text(encoding="utf-8"))['details'], key=lambda r:r['valid_index'])
with (ROOT/'outputs/q2_prediction/q2_systematic/validation_strength_targets.csv').open(encoding='utf-8-sig') as f:
    targets = list(csv.DictReader(f))
assert [r['valid_index'] for r in rows] == [int(r['valid_index']) for r in targets] == list(range(728))
assert [r['sample_id'] for r in rows] == [r['sample_id'] for r in targets]
y = np.array([float(r['true_strength']) for r in targets])
p = np.array([r['strength'] for r in rows])
c = np.array([int(r['true_polarity']) for r in targets])
ref = json.loads((ROOT/'outputs/q2_prediction/q2_systematic/gate_recon_ensemble.json').read_text(encoding="utf-8"))['complete']
assert np.isclose(np.abs(p-y).mean(), ref['mae_consistent'], atol=1e-6)
assert np.isclose(np.corrcoef(y,p)[0,1], ref['pearson_consistent'], atol=1e-6)

fig, (a,b) = plt.subplots(1,2,figsize=(8.7,4.0),gridspec_kw={'width_ratios':[1,1.08]})
fig.subplots_adjust(left=.075,right=.935,bottom=.16,top=.87,wspace=.43)
colors=[BLUE, TEAL, GOLD]
names=['真实负向','真实中性','真实正向']
for k,col in enumerate(colors):
    mask=c==k
    a.scatter(y[mask],p[mask],s=10,alpha=.38,color=col,edgecolors='none',rasterized=False)
a.plot([-3,3],[-3,3],color='#818995',ls=(0,(4,3)),lw=.85,zorder=0)
a.set(xlim=(-3.15,3.15),ylim=(-3.15,3.15),xlabel='真实情感强度',ylabel='预测情感强度')
a.set_xticks(range(-3,4));a.set_yticks(range(-3,4));a.set_aspect('equal',adjustable='box')

# Fixed unit-width intervals, keeping exact zero as its own neutral group.
masks=[(y>=-3)&(y<-2),(y>=-2)&(y<-1),(y>=-1)&(y<0),y==0,(y>0)&(y<=1),(y>1)&(y<=2),(y>2)&(y<=3)]
labels=['[−3, −2)','[−2, −1)','[−1, 0)','0','(0, 1]','(1, 2]','(2, 3]']
stats=[]
all_residual=p-y
limit=np.ceil(np.abs(all_residual).max()*2)/2+.15
b.axvline(0,color='#818995',ls=(0,(4,3)),lw=.85)
for i,(mask,label) in enumerate(zip(masks,labels)):
    residual=p[mask]-y[mask]
    q1,med,q3=np.quantile(residual,[.25,.5,.75])
    col=colors[0 if i<3 else 1 if i==3 else 2]
    x=np.linspace(residual.min(),residual.max(),240)
    density=gaussian_kde(residual)(x)
    half=.33*density/density.max()
    b.fill_between(x,i-half,i+half,color=col,alpha=.16,lw=0)
    b.plot(x,i-half,color=col,lw=.7,alpha=.8)
    b.plot(x,i+half,color=col,lw=.7,alpha=.8)
    b.hlines(i,q1,q3,color=col,lw=3,alpha=.95)
    b.plot(med,i,'o',color=col,ms=4.4,mec='white',mew=.8)
    b.text(1.025,i,str(mask.sum()),transform=b.get_yaxis_transform(),ha='left',va='center',fontsize=8)
    stats.append({'true_strength_interval':label,'n':int(mask.sum()),'mae':float(np.abs(residual).mean()),'median_residual':float(med),'residual_q25':float(q1),'residual_q75':float(q3)})
b.text(1.025,-.72,'样本数',transform=b.get_yaxis_transform(),ha='left',va='center',fontsize=8)
b.set_yticks(range(7),labels)
b.set(xlim=(-limit,limit),ylim=(6.65,-1.05),xlabel='预测强度 − 真实强度',ylabel='真实情感强度区间')
b.locator_params(axis='x',nbins=5)
b.set_axisbelow(True)
b.grid(axis='x',color='#E8EBEF',lw=.5)
for ax in (a,b):
    ax.spines[['top','right']].set_visible(False)
    ax.tick_params(length=3)
handles=[Line2D([],[],marker='o',ls='none',color=col,ms=5,label=name) for col,name in zip(colors,names)]
handles += [Line2D([],[],color='#818995',ls='--',lw=.85,label='理想预测')]
fig.legend(handles=handles,loc='upper center',ncol=4,frameon=False,columnspacing=2)
prepare(fig)
out=ROOT/'paper/figures/情感强度预测与分区误差.pdf'
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out,bbox_inches='tight',pad_inches=.1)
plt.close(fig)
result={'source':'attachment2 valid; frozen gate_recon ensemble; sign-projected strength','n':len(rows),'mae':float(np.abs(p-y).mean()),'pearson':float(np.corrcoef(y,p)[0,1]),'intervals':stats,'interval_mark':'dot = median; thick segment = interquartile range, not confidence interval'}
(ROOT/'outputs/q2_prediction/q2_systematic/strength_diagnostics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n', encoding="utf-8")
print(json.dumps(result,ensure_ascii=False,indent=2))
print(out)
