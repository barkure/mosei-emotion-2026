# 本程序及代码是在人工智能工具辅助下完成的
# 人工智能工具名称：DeepSeek，型号：DeepSeek V4.1 Flash，开发公司：杭州深度求索人工智能基础技术研究有限公司，版本颁布日期：2026-09-10

"""Synchronize the paper's plotting-software PDFs to the gallery.

Run after plotting. TeX/TikZ figures stay in paper/ and are not exported here.
"""
from __future__ import annotations

import shutil
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper"
OUT = ROOT / "outputs/figures"

PAPER_FIGURES = {
    "数据概览.pdf": "图 3.1 数据概览.pdf",
    "音画时间基准.pdf": "图 4.1 音画时间基准核验.pdf",
    "音文筛查分布.pdf": "图 4.2 音文核查与词时间覆盖.pdf",
    "词时间筛选.pdf": "图 4.3 词时间筛选与处理去向.pdf",
    "模态来源概览.pdf": "图 4.4 词级覆盖率累积分布.pdf",
    "典型样本对齐.pdf": "图 4.5 典型样本对齐.pdf",
    "混合专家模型框架.pdf": "图 5.1 混合专家模型框架.pdf",
    "极性预测流向.pdf": "图 5.2 极性预测流向.pdf",
    "情感强度预测与分区误差.pdf": "图 5.4 情感强度预测与分区误差.pdf",
    "文本缺失位置效应.pdf": "图 5.5 文本缺失位置效应.pdf",
    "预测置信度可靠性.pdf": "图 5.3 置信度可靠性图.pdf",
    "模态移除敏感性.pdf": "图 5.6 模态移除敏感性.pdf",
    "缺失条件下的门控权重分布.pdf": "图 5.7 缺失条件下的门控权重分布.pdf",
    "预测解释流程.pdf": "图 6.1 预测解释流程.pdf",
    "预测类别模态贡献分布.pdf": "图 6.2 按预测类别贡献分布.pdf",
    "文本局部窗口响应.pdf": "图 6.3 三十条文本窗口敏感热图.pdf",
    "局部解释忠实度散点.pdf": "图 6.4 局部解释忠实度散点.pdf",
    "解释样本.pdf": "图 6.6 解释样本.pdf",
    "预测正确性与解释响应分布.pdf": "图 6.5 验证集预测正确性与解释响应分布.pdf",
}

def sync():
    with (ROOT / "docs/图表来源.csv").open(encoding="utf-8-sig", newline="") as f:
        records = list(csv.DictReader(f))
    indexed_pdfs = {row['output'] for row in records if row['output'].endswith('.pdf')}
    expected_pdfs = {f'outputs/figures/{name}' for name in PAPER_FIGURES.values()}
    assert indexed_pdfs == expected_pdfs, "图表来源索引与图库映射不一致"
    for source_name in PAPER_FIGURES:
        source = PAPER / "figures" / source_name
        assert source.is_file(), source
    OUT.mkdir(parents=True, exist_ok=True)
    for source_name, target_name in PAPER_FIGURES.items():
        source = PAPER / "figures" / source_name
        shutil.copyfile(source, OUT / target_name)
    print(f"已同步 {len(PAPER_FIGURES)} 张论文绘图 PDF 到 {OUT.relative_to(ROOT)}")


if __name__ == '__main__':
    sync()
