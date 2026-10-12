# 本程序及代码是在人工智能工具辅助下完成的
# 人工智能工具名称：DeepSeek，型号：DeepSeek V4.1 Flash，开发公司：杭州深度求索人工智能基础技术研究有限公司，版本颁布日期：2026-09-10

"""Reproduce the 19 published plots from fixed outputs and one original clip."""
from pathlib import Path
import argparse
import csv
import shutil
import subprocess
import sys

from sync_paper_gallery import PAPER_FIGURES

ROOT = Path(__file__).resolve().parents[2]
Q1 = 'outputs/q1_features/'
Q2 = 'outputs/q2_prediction/q2_systematic/'
Q3 = 'outputs/q3_explanation/q3/'
RESULTS = Q3 + 'validation_explanation_all.json'
JOBS = [
    ('generate_dataset_figures.py', ['3.1'], ['outputs/data_audit/attachment2_audit.json']),
    ('plot_problem1_overview.py', ['4.1', '4.2', '4.3', '4.4', '4.5'],
     [Q1+'attachment1_timebase/summary.csv', Q1+'q1_final/word_alignment_summary.csv',
      Q1+'attachment1_alignment/recovered/*.json', Q1+'q1_final/features/-egA8-b7-3M__9.npz',
      'data/attachment1/-egA8-b7-3M/9.mp4']),
    ('plot_model_evidence.py', ['5.1', '5.6'], [Q2+'gate_recon_ensemble.json']),
    ('plot_prediction_patterns.py', ['5.2', '5.5'], [Q2+'gate_recon_ensemble.json', RESULTS]),
    ('plot_gallery_diagnostics.py', ['5.3', '6.4'], [Q2+'gate_recon_ensemble.json', RESULTS]),
    ('plot_strength_diagnostics.py', ['5.4'], [Q2+'validation_strength_targets.csv', Q2+'gate_recon_ensemble.json', RESULTS]),
    ('plot_mechanism_diagnostics.py', ['5.7', '6.5'], [Q2+'gate_response_diagnostics.json', RESULTS]),
    ('plot_explanation_flow.py', ['6.1'], []),
    ('plot_explanation_overview.py', ['6.2', '6.3'], [RESULTS]),
    ('../q3_explanation/plot_q3_explanation_cases.py', ['6.6'], [Q3+'attachment4_explanations_full.json']),
]


def source_rows():
    rows = [dict(figure='1.1', output='paper/figures/总体技术路线.tex',
                 source='paper/figures/总体技术路线.tex', inputs='论文方法定义',
                 command='由LaTeX/TikZ随论文排版', external_requirements='LaTeX/TikZ及论文字体')]
    for script, numbers, inputs in JOBS:
        source = str((ROOT/'src/figures'/script).resolve().relative_to(ROOT))
        for number in numbers:
            pdf, gallery = next((a, b) for a, b in PAPER_FIGURES.items() if b.startswith(f'图 {number} '))
            rows.append(dict(figure=number, output='outputs/figures/'+gallery, source=source,
                             inputs=';'.join(inputs), command='pixi run python '+source,
                             external_requirements='SimSun、Times New Roman'+('；原始视频与FFmpeg' if number.startswith('4.') else '')))
    return sorted(rows, key=lambda r: tuple(map(int, r['figure'].split('.'))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--all', action='store_true', help='生成全部19张PDF并同步图库')
    parser.add_argument('--index', action='store_true', help='更新docs/图表来源.csv')
    args = parser.parse_args()
    rows = source_rows()
    if args.index:
        target = ROOT/'docs/图表来源.csv'
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print('已更新20幅图的来源索引。')
    if not args.all:
        for row in rows:
            print(row['figure'], row['source'])
        return
    missing = sorted({p for _, _, inputs in JOBS for p in inputs if not list(ROOT.glob(p))})
    if missing:
        parser.error('缺少输入：'+'、'.join(missing))
    (ROOT/'paper/figures').mkdir(parents=True, exist_ok=True)
    gallery = ROOT/'outputs/figures'
    gallery.mkdir(parents=True, exist_ok=True)
    for script, numbers, _ in JOBS:
        print('生成图 '+', '.join(numbers), flush=True)
        subprocess.run([sys.executable, str(ROOT/'src/figures'/script)], cwd=ROOT, check=True)
    for source, target in PAPER_FIGURES.items():
        shutil.copy2(ROOT/'paper/figures'/source, gallery/target)
    print('已生成19张绘图PDF；图1.1的TikZ源文件单独保留。')


if __name__ == '__main__':
    main()
