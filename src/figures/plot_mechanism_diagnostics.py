# 本程序及代码是在人工智能工具辅助下完成的
# 人工智能工具名称：ChatGPT，型号：GPT-6 Astra，开发公司：OpenAI，版本颁布日期：2026-09-04
"""Mechanism and error analyses from fixed validation results only."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from figure_style import use_serif, prepare, BLUE, TEAL, GOLD, RED

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'paper/figures'
Q2=ROOT/'outputs/q2_prediction/q2_systematic'
use_serif(ROOT)
plt.rcParams.update({'font.size':9, 'axes.linewidth':.65, 'xtick.major.width':.65,'ytick.major.width':.65})

def save(fig,name):
    prepare(fig)
    fig.savefig(OUT/name,bbox_inches='tight',pad_inches=.10)
    plt.close(fig)

def gate_cloud():
    report=json.loads((Q2/'gate_response_diagnostics.json').read_text(encoding="utf-8"))
    names=['完整输入','文本中部缺失','音频中部缺失','视频中部缺失']
    colors=['#637084',BLUE,TEAL,GOLD]
    keys=['complete','text','audio','vision']
    fig,(left,ax)=plt.subplots(1,2,figsize=(8.3,3.8),sharey=True,gridspec_kw={'width_ratios':[.6,7]})
    fig.subplots_adjust(left=.16,right=.92,bottom=.18,top=.97,wspace=.07)
    rng=np.random.default_rng(2026)
    for i,(key,col) in enumerate(zip(keys,colors)):
        v=np.array(report['scenarios'][key]['sample_mean'])
        assert np.all((v==0)|(v>=.50))
        positive=v[v>0]
        x=np.linspace(max(.50,positive.min()-.02),min(1,positive.max()+.02),400)
        density=gaussian_kde(positive)(x)
        upper=i-.10-.32*density/density.max()
        ax.fill_between(x,upper,i-.10,color=col,alpha=.24,lw=0)
        ax.plot(x,upper,color=col,lw=.9)
        jitter=i+.13+rng.uniform(0,.19,len(v))
        for target in (left,ax):
            target.scatter(v,jitter,s=3,color=col,alpha=.25,edgecolors='none')
        if (v==0).any():
            left.text(0,i-.12,f'{(v==0).sum()} 条',ha='center',va='center',fontsize=8,color=col)
        q1,med,q3=np.quantile(v,[.25,.5,.75])
        ax.plot([q1,q3],[i,i],color=col,lw=3.5,solid_capstyle='round')
        ax.plot(med,i,'o',ms=4,color=col,mec='white',mew=.7)
        ax.text(1.025,i,f'{med:.4f}',transform=ax.get_yaxis_transform(),va='center',ha='left',fontsize=8)
    ax.text(1.025,-.57,'中位数',transform=ax.get_yaxis_transform(),ha='left',va='center',fontsize=8)
    ax.set(ylim=(3.55,-.65),xlim=(.50,1.0));left.set_xlim(-.025,.025)
    left.set_yticks(range(4),names);left.set_xticks([0]);ax.set_xticks(np.arange(.5,1.01,.1))
    ax.tick_params(axis='y',labelleft=False)
    for target in (left,ax):
        target.spines[['top','right','left']].set_visible(False);target.tick_params(axis='y',length=0,pad=9)
        target.grid(axis='x',lw=.5,color='#E4E7EB');target.set_axisbelow(True)
    for target,x in [(left,1),(ax,0)]:
        target.plot([x-.009,x+.009],[-.015,.015],transform=target.transAxes,color='black',lw=.7,clip_on=False)
    fig.text(.56,.035,'文本专家权重',ha='center')
    save(fig,'缺失条件下的门控权重分布.pdf')

def explanation_ecdf():
    rows=json.loads((ROOT/'outputs/q3_explanation/q3/validation_explanation_all.json').read_text(encoding="utf-8"))['details']
    fig,axs=plt.subplots(1,2,figsize=(8.3,3.35),sharey=True)
    fig.subplots_adjust(left=.085,right=.98,bottom=.2,top=.84,wspace=.25)
    summary={}
    for correct,col,name in [(True,BLUE,'预测正确'),(False,RED,'预测错误')]:
        part=[r for r in rows if r['prediction_correct']==correct]
        phi=np.array([r['shapley_probability'][0] for r in part])
        gain=np.array([r['wider_window_transfer']['text']['high']-r['wider_window_transfer']['text']['random'] for r in part])
        summary[name]={'n':len(part),'positive_text_phi':int((phi>0).sum()),'median_text_phi':float(np.median(phi)),
            'median_wider_gain':float(np.median(gain)), 'mean_wider_gain':float(gain.mean()),'positive_wider_gain':int((gain>0).sum())}
        for ax,v in zip(axs,[phi,gain]):
            x=np.sort(v);y=np.arange(1,len(v)+1)/len(v)
            ax.step(x,y,where='post',color=col,lw=1.5,label=f'验证集{name}   n = {len(v)}')
    for ax,xlabel in zip(axs,['文本 Shapley 概率贡献','高分窗与随机窗的概率下降差']):
        ax.axvline(0,color='#939BA5',lw=.75,ls=(0,(3,3)),zorder=0)
        ax.set(xlabel=xlabel,ylim=(0,1));ax.set_yticks([0,.25,.5,.75,1]);ax.grid(axis='y',lw=.5,color='#E7E9ED');ax.locator_params(axis='x',nbins=5)
    axs[0].set_ylabel('样本累计比例')
    fig.legend(*axs[0].get_legend_handles_labels(),loc='upper center',ncol=2,frameon=False,columnspacing=3)
    save(fig,'预测正确性与解释响应分布.pdf')
    target=ROOT/'outputs/q3_explanation/q3/explanation_response_diagnostics.json'
    target.write_text(json.dumps({'source':'attachment2 validation; fixed ensemble; 40% wider-window transfer','groups':summary},ensure_ascii=False,indent=2)+'\n', encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':
    gate_cloud();explanation_ecdf()
