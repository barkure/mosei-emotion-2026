# 问题三：可复核的预测解释

沿用问题二固定的 `gate_recon` 三种子集成，在附件二validation的728条样本上检查解释机制，再对附件四20条样本推理。

| 脚本 | 作用与产物（结果均位于`outputs/q3_explanation/q3/`） |
| --- | --- |
| `explain_q3_gate_ensemble.py` | 八组合精确Shapley与20%滑窗遮挡；输出`validation_explanation_all.json`、`attachment4_explanations_full.json`及专项CSV |
| `align_q3_evidence_time.py` | 对各自原视频生成CTC文本时间候选及05邻近画面；输出`attachment4_word_alignment_candidates.json`、`visual_candidate_05.png`、manifest与核验表 |
| `merge_review_status.py` | 把已归档回看反馈合入CSV、完整JSON、核验表与manifest；时间候选重跑后执行 |
| `export_paper_tables.py` | 从完整JSON导出预测贡献与关键证据两张TeX表至`paper/tables/` |
| `plot_q3_explanation_cases.py` | 从固定结果生成`paper/figures/解释样本.pdf` |

共享依赖为问题二模型定义与权重、`common.prep`和`q1_features.align_words`。环境与数据准备见[项目运行](../../README.md#项目运行)。

## 验证与媒体定位

以下命令在项目根目录执行。全量验证使用附件二 `aligned_50.pkl`：

```powershell
pixi run python src/q3_explanation/explain_q3_gate_ensemble.py --stage validation --full-validation
```

附件四核心预测与特征位置解释无需 FFmpeg。若需重建候选时段和截图，先完成附件四推理，准备 `data/attachment4/aligned/videos/` 下的原视频，安装 FFmpeg（含 ffprobe）并加入 PATH，再执行：

```powershell
pixi run python src/q3_explanation/align_q3_evidence_time.py
pixi run python src/q3_explanation/merge_review_status.py
```

7条文本候选时段与05邻近画面的反馈分别保存在结果目录的 `text_time_user_review_20260925.json` 和 `video05_user_review_20260925.json`，已合入解释结果。反馈支持内容覆盖与邻近画面核对；精确词边界误差和视频特征到原始帧的映射尚未验证。方法、贡献定义与媒体对应范围见[解释与核验](../../docs/问题三解释与核验.md)。
