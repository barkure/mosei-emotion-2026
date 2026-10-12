# 问题二：局部模态缺失下的情感预测

固定使用附件二 `aligned_50.pkl`，train拟合、valid选型，附件三仅最终推理。正式模型为种子2026、2027、2028的文本表征引导MoE（`gate_recon`）集成。

| 脚本 | 作用与产物 |
| --- | --- |
| `experiment_q2_mask_fusion.py` | 训练五种`--mode`：`av`、`av_lstm`、`gate`、`gate_recon`、`cross_attn`；`--mode gate --neutral-loss-weight 0.3`产生第六类候选`gate_neutral`。权重、训练历史和验证记录写入`outputs/q2_prediction/q2_systematic/` |
| `evaluate_q2_systematic_ensemble.py` | 三种子分类logits与原始强度分别平均，再作强度符号投影；验证输出`<mode>_ensemble.json` |
| `infer_attachment3_q2_ensemble.py` | 附件三30条推理；输出`attachment3_predictions_gate_recon_ensemble.csv`及manifest |
| `analyze_gate_responses.py` | 附件二验证集逐样本门控诊断；输出`q2_systematic/gate_response_diagnostics.json` |

共享依赖为 `common.prep`（输入、冻结BERT、缺失增强与强度投影）及 `common.encoders.AVEncoder`。仓库保留正式三种子权重，其他候选保留数值记录。

使用固定权重验证与推理：

```bash
pixi run python src/q2_prediction/evaluate_q2_systematic_ensemble.py --mode gate_recon --split valid --scenarios
pixi run infer-attachment3-aligned
```

728条完整验证输入的ACC、Macro-F1、投影后MAE、Pearson分别为0.6484、0.6265、0.5769、0.6463；缺失评价包含18种局部场景及3种整模态剔除压力场景。模型结构与结果口径见[方法说明](../../docs/问题二模型与实验.md)。

## 三种子重训

按[项目运行](../../README.md#项目运行)准备环境、公开模型和附件二 `aligned_50.pkl`。以下命令在项目根目录执行，会覆盖同名权重，请在独立副本中运行：

```powershell
foreach ($seed in 2026, 2027, 2028) {
    pixi run python src/q2_prediction/experiment_q2_mask_fusion.py --mode gate_recon --seed $seed --epochs 12 --batch-size 64
}
```
