# 本程序及代码是在人工智能工具辅助下完成的
# 人工智能工具名称：DeepSeek，型号：DeepSeek V4.1 Flash，开发公司：杭州深度求索人工智能基础技术研究有限公司，版本颁布日期：2026-09-10

"""Export appendix tables and code excerpts from fixed results and actual source."""
from pathlib import Path
import ast
import csv
import json
import pickle
import sys
import textwrap

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src/q1_features'))
sys.path.insert(0, str(ROOT / 'src/q3_explanation'))
from audio_features import FEATURE_NAMES
from feature_extractors import BLENDSHAPE_NAMES_52
from export_paper_tables import table

Q1 = ROOT / 'outputs/q1_features/q1_final'
Q2 = ROOT / 'outputs/q2_prediction/q2_systematic'
Q3 = ROOT / 'outputs/q3_explanation/q3'
TABLES = ROOT / 'paper/tables'
CODE = ROOT / 'paper/code'


def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def feature_dictionary():
    files = sorted((Q1 / 'features').glob('*.npz'))
    assert len(files) == 100
    for path in files:
        with np.load(path, allow_pickle=False) as data:
            assert data['audio_feature_names'].tolist() == FEATURE_NAMES
            assert data['blendshape_names'].tolist() == BLENDSHAPE_NAMES_52
            assert data['word_audio_features'].shape[1] == 17
            assert data['word_video_features'].shape[1] == 52
    rows = []
    audio = [
        ('MFCC', '对数mel谱的正交倒谱系数', '无量纲'),
    ] * 13 + [
        ('对数RMS', '去均值波形均方根幅值的自然对数', '无量纲'),
        ('基频', '自相关峰值对应的周期频率', 'Hz'),
        ('频谱质心', '幅值谱的频率加权均值', 'Hz'),
        ('过零率', '相邻采样点符号变化次数除以399', '无量纲'),
    ]
    for i, (name, (group, meaning, unit)) in enumerate(zip(FEATURE_NAMES, audio)):
        rows.append(dict(modality='audio', index_zero_based=i, column_one_based=i+1,
                         feature_name=name, feature_group=group, meaning=meaning, unit=unit,
                         native_valid_mask='audio_window_pitch_observed' if i == 14 else 'audio_window_observed',
                         word_valid_mask='word_pitch_observed' if i == 14 else 'word_audio_observed'))
    groups = [('_neutral', '基准系数'), ('brow', '眉部'), ('cheek', '面颊'),
              ('eye', '眼部'), ('jaw', '下颌'), ('mouth', '口部'), ('nose', '鼻部')]
    for i, name in enumerate(BLENDSHAPE_NAMES_52):
        group = next(group for prefix, group in groups if name.startswith(prefix))
        rows.append(dict(modality='video', index_zero_based=i, column_one_based=i+1,
                         feature_name=name, feature_group=group, meaning='Face Landmarker形变系数', unit='无量纲',
                         native_valid_mask='vision_frame_single_face_valid', word_valid_mask='word_video_observed'))
    write_csv(Q1 / 'feature_dimensions.csv', rows)
    text = r'''\begin{table}[htbp]
\centering
\caption{17维声学特征的排列与有效性}\label{tab:appendix-audio-dimensions}
\begin{tblr}{width=\linewidth,colspec={c c c X[c] X[c]},cells={valign=m,font=\normalsize\songti},
 colsep=4pt,rowsep=3pt,hline{1,7}={0.8pt},hline{2}={0.4pt}}
列序号 & 特征 & 维数 & 定义 & 可用条件 \\
1--13 & MFCC $c_0$--$c_{12}$ & 13 & 26通道对数mel谱的前13个正交倒谱系数 & 存在完整声学窗 \\
14 & 对数RMS & 1 & 去均值波形均方根幅值的自然对数 & 存在完整声学窗 \\
15 & 基频/Hz & 1 & 60--400 Hz内自相关峰值对应频率 & 峰值至少0.30，RMS至少$10^{-3}$ \\
16 & 频谱质心/Hz & 1 & 幅值谱的频率加权均值 & 完整声学窗；全零谱取零 \\
17 & 过零率 & 1 & 相邻采样点符号变化次数除以399 & 存在完整声学窗 \\
\end{tblr}
\end{table}
'''
    (TABLES / '声学特征维度.tex').write_text(text, encoding="utf-8")
    grouped = []
    for prefix, name in groups:
        indexes = [i+1 for i, n in enumerate(BLENDSHAPE_NAMES_52) if n.startswith(prefix)]
        assert indexes == list(range(indexes[0], indexes[-1]+1))
        span = str(indexes[0]) if len(indexes) == 1 else f'{indexes[0]}--{indexes[-1]}'
        grouped.append([span, name, str(len(indexes))])
    (TABLES / '面部特征分组.tex').write_text(
        '\\begin{table}[htbp]\n\\centering\n'
        '\\caption{52维面部系数的分组与排列}\\label{tab:appendix-video-dimensions}\n'
        '\\begin{tabular*}{\\linewidth}{@{\\extracolsep{\\fill}}ccc@{}}\n\\toprule\n'
        '列序号 & 系数组 & 维数 \\\\\n\\midrule\n'
        + '\n'.join(' & '.join(row)+r' \\' for row in grouped)
        + '\n\\bottomrule\n\\end{tabular*}\n\\end{table}\n', encoding="utf-8")


def strength_contributions():
    data = json.loads((Q3 / 'attachment4_explanations_full.json').read_text(encoding="utf-8"))
    assert [r['sample_id'] for r in data] == [f'{i:02d}' for i in range(1, 21)]
    rows, tex = [], []
    for r in data:
        values = np.asarray(r['shapley_strength_raw'], dtype=float)
        assert values.shape == (3,) and np.isfinite(values).all()
        main = ['文本', '音频', '视频'][int(np.abs(values).argmax())]
        rows.append(dict(sample_id=r['sample_id'], text=float(values[0]), audio=float(values[1]),
                         video=float(values[2]), main_modality=main))
        tex.append([r['sample_id'], *[f'${v:.4f}$' for v in values], main])
    write_csv(Q3 / 'attachment4_strength_contributions.csv', rows)
    (TABLES / '附件四强度贡献.tex').write_text(table(
        '附件四全部样本的原始强度贡献', 'tab:appendix-strength-contributions', 'ccccc',
        r'编号 & 文本贡献 & 音频贡献 & 视频贡献 & 强度主模态 \\', tex, 5), encoding="utf-8")


def model_components():
    specs = [
        ('av', '音视频卷积', '64维残差卷积', '音视频拼接', '无', '无', [2026]),
        ('av_lstm', '音视频BiLSTM', '双向LSTM', '音视频拼接', '无', '无', [2026]),
        ('cross_attn', '掩码跨模态注意力', '64维残差卷积', '两层跨模态注意力', '无', '无', [2026]),
        ('gate', '基础MoE', '64维残差卷积', '样本级双专家门控', '文本0.2；音视频0.3', '无', [2026, 2027, 2028]),
        ('gate_neutral', '中性辅助MoE', '64维残差卷积', '样本级双专家门控', '文本0.2；音视频0.3', '中性二元损失0.3', [2026, 2027]),
        ('gate_recon', '文本表征引导MoE', '64维残差卷积', '样本级双专家门控', '文本0.2；音视频0.3', '文本重构0.02', [2026, 2027, 2028]),
    ]
    rows = []
    for mode, name, encoder, fusion, branch, auxiliary, seeds in specs:
        epochs = []
        for seed in seeds:
            result = json.loads((Q2 / f'{mode}_seed{seed}.json').read_text(encoding="utf-8"))
            assert result['seed'] == seed and result['data_version'] == 'aligned_50'
            epochs.append(result['selected_epoch'])
        rows.append(dict(mode=mode, model=name, av_encoder=encoder, fusion=fusion,
                         branch_loss=branch, auxiliary_loss=auxiliary,
                         seeds=';'.join(map(str, seeds)), selected_epochs=';'.join(map(str, epochs))))
    write_csv(Q2 / 'candidate_components.csv', rows)


def plotting_targets():
    with (ROOT / 'data/attachment2/aligned_50.pkl').open('rb') as f:
        valid = pickle.load(f)['valid']
    results = sorted(json.loads((Q3 / 'validation_explanation_all.json').read_text(encoding="utf-8"))['details'],
                     key=lambda r: r['valid_index'])
    assert len(results) == len(valid['id']) == 728
    labels = np.asarray(valid['classification_labels']).ravel()
    strengths = np.asarray(valid['regression_labels']).ravel()
    rows = []
    for i, r in enumerate(results):
        assert r['valid_index'] == i and r['sample_id'] == valid['id'][i]
        assert r['true_polarity'] == labels[i]
        rows.append(dict(valid_index=i, sample_id=r['sample_id'],
                         true_polarity=int(labels[i]), true_strength=float(strengths[i])))
    write_csv(Q2 / 'validation_strength_targets.csv', rows)


def code_excerpts():
    source = (ROOT / 'src/q1_features/realign_reviewed_words.py').read_text(encoding="utf-8")
    tree = ast.parse(source)
    funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    helper = ast.parse((ROOT / 'src/q1_features/align_words.py').read_text(encoding="utf-8"))
    minimum = next(n for n in helper.body if isinstance(n, ast.FunctionDef) and n.name == 'min_required_frames')
    def listing(name, nodes, prefix=''):
        content = '\n\n'.join(ast.unparse(n) for n in nodes)
        (CODE / name).write_text('% Generated from the actual Q1 source; formatting only.\n'
                                '\\begin{lstlisting}\n' + prefix + content + '\n\\end{lstlisting}\n', encoding="utf-8")
    listing('ctc_path.tex', [minimum, funcs['ctc_path']])
    listing('word_boundaries.tex', [funcs['targets_for'], funcs['align']], 'STEP = 0.02\n\n')
    block = source[source.index('    for block in blocks:'):source.index('   result=dict(stable_id=')]
    nodes = ast.parse(textwrap.dedent(block)).body
    listing('local_word_filter.tex', nodes, 'CONFIDENCE = 0.50\nSTABILITY = 0.10\n\n')


if __name__ == '__main__':
    TABLES.mkdir(parents=True, exist_ok=True)
    CODE.mkdir(parents=True, exist_ok=True)
    feature_dictionary()
    strength_contributions()
    model_components()
    plotting_targets()
    code_excerpts()
    print('Exported 69 feature dimensions, 20 strength contributions, 6 model configurations and 3 code excerpts.')
