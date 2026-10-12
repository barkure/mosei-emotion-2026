# 本程序及代码是在人工智能工具辅助下完成的
# 人工智能工具名称：ChatGPT，型号：GPT-6 Astra，开发公司：OpenAI，版本颁布日期：2026-09-04

"""Explanation procedure of the fixed Q3 model; outputs a vector PDF."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib import patheffects
from figure_style import use_serif, BLUE, TEAL, INK

ROOT = Path(__file__).resolve().parents[2]

def main():
    use_serif(ROOT)
    fig, ax = plt.subplots(figsize=(7.8, 5.1))
    ax.set(xlim=(-.12, 9.72), ylim=(-1.15, 6.12))
    ax.axis('off')
    def box(x, y, w, h, title, lines, color):
        ax.add_patch(Rectangle((x,y),w,h,facecolor='white',edgecolor='#C7CED8',lw=.75))
        ax.plot([x,x+w],[y+h,y+h],color=color,lw=1.8,solid_capstyle='butt')
        texts=[title,*lines]
        step=.29
        top=y+h/2+step*(len(texts)-1)/2
        for i, text in enumerate(texts):
            artist=ax.text(x+w/2,top-i*step,text,ha='center',va='center',fontsize=9.5 if i==0 else 9,color='black')
            if i==0:
                artist.set_path_effects([patheffects.withStroke(linewidth=.18,foreground='black')])
    def arrow(x,y0,y1,color):
        ax.add_patch(FancyArrowPatch((x,y0),(x,y1),arrowstyle='->',mutation_scale=10,
                                    color=color,lw=.9,shrinkA=3,shrinkB=3))
    box(0,5.05,9.6,.95,'固定三种子集成模型',
        ['完整输入预测极性与强度，后续遮挡均考察该预测类别的概率'],INK)
    box(0,3.55,4.4,1.0,'八种模态保留组合',
        ['枚举文本、音频、视频的全部子集','遮挡时同步更新输入与观测掩码'],BLUE)
    box(5.2,3.55,4.4,1.0,'逐模态滑窗遮挡',
        ['窗口覆盖该模态20%的已观测位置','逐窗遮挡并重新计算预测'],TEAL)
    box(0,2.10,4.4,1.0,'精确 Shapley 分解',
        ['计算类别概率的有符号模态贡献','另对原始回归强度进行贡献分解'],BLUE)
    box(5.2,2.10,4.4,1.0,'计算局部概率下降',
        ['完整输入概率 − 遮挡后概率','保存各模态全部窗口的变化'],TEAL)
    box(0,.65,4.4,1.0,'确定主要参考模态',
        ['取类别概率有符号贡献最大的模态','同时保留三模态贡献与正贡献占比'],BLUE)
    box(5.2,.65,4.4,1.0,'提取局部关键窗口',
        ['各模态选取概率下降最大的窗口','保留位置、原文及遮挡响应'],TEAL)
    for x,color in [(2.2,BLUE),(7.4,TEAL)]:
        for y0,y1 in [(5.05,4.55),(3.55,3.10),(2.10,1.65),(.65,.15)]:
            arrow(x,y0,y1,color)
    box(0,-1.05,9.6,1.2,'主要参考模态的关键证据与来源记录',
        ['词元映射到原文，时间与视频帧附核验状态',
         '未恢复媒体时间的窗口保留特征位置'],INK)
    fig.subplots_adjust(left=.025,right=.975,bottom=.025,top=.975)
    path=ROOT/'paper/figures/预测解释流程.pdf'
    fig.savefig(path,bbox_inches='tight',pad_inches=.08)
    plt.close(fig)
    print(path)

if __name__=='__main__':
    main()
