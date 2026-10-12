# MOSEI 多模态情感预测

2026 年中国研究生数学建模竞赛 E 题的代码与成果归档，涵盖多模态时序对齐、局部缺失下的情感预测和可解释性分析。

[论文 PDF](paper/main.pdf) · [方法文档](docs/README.md) · [论文图表](outputs/figures/) · [数据放置说明](data/README.md)

## 方法与结果

| 任务 | 方法 | 已完成结果 |
| --- | --- | --- |
| 问题一 | CTC 词时间筛选与区间交叠聚合 | 100 份自包含特征，1533 个词采用时间戳 |
| 问题二 | 文本表征引导 MoE，三种子集成 | 728 条验证，18 种局部缺失场景，附件三 30 条预测 |
| 问题三 | Shapley 模态贡献与滑窗遮挡 | 728 条验证解释，附件四 20 条预测与解释 |

固定模型 `gate_recon` 在附件二完整验证输入上的结果：

| Accuracy | Macro-F1 | MAE | Pearson |
| :---: | :---: | :---: | :---: |
| 0.6484 | 0.6265 | 0.5769 | 0.6463 |

以上为附件二验证集指标，附件三、四仅用于最终推理；详细方法见[文档目录](docs/README.md)。

## 项目运行

环境为 **Windows x64 + Pixi + Python 3.12**，支持 CUDA 或 CPU。按 [data/README.md](data/README.md) 放置赛题数据，以下命令均在项目根目录执行：

```powershell
pixi install --locked
pixi run resources

# 验证已保存的问题一特征
pixi run python src/q1_features/verify_final_word_features.py

# 附件三、四预测与解释
pixi run infer-attachment3-aligned
pixi run python src/q3_explanation/explain_q3_gate_ensemble.py --stage attachment4
```

| 目录 | 内容与详细入口 |
| --- | --- |
| `src/common/` | [数据接口与掩码规则](src/common/README.md) |
| `src/q1_features/` | [特征提取与完整重建命令](src/q1_features/README.md) |
| `src/q2_prediction/` | [训练、验证与专项推理](src/q2_prediction/README.md) |
| `src/q3_explanation/` | [解释与可选媒体定位](src/q3_explanation/README.md) |
| `src/figures/` | [论文绘图](src/figures/README.md) |
| `models/` | [公开预训练资源放置说明](models/README.md)，模型文件不随仓库分发 |
| `outputs/` | 正式特征、训练权重、实验结果与回看记录 |
| `paper/`、`docs/` | 已提交论文、方法说明及核验依据 |
