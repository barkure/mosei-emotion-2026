# 问题一：三模态特征提取与词时间对齐

方法为基于词时间戳与区间交叠的三模态时序对齐。全部100条样本保留实际长度、官方词序与原生序列；80条中1533词有采用时间，1374词有三模态观测。模型参数、采用规则和证据范围见[方法说明](../../docs/问题一特征与对齐.md)。

## 入口与模块

| 文件 | 作用 |
| --- | --- |
| `attachment1_inventory.py` | 样本、原始文件与标签清单 |
| `verify_timebase_attachment1.py` | 解码音视频时间基准 |
| `extract_native_features.py` | 官方文本BERT编码及原生声学短窗特征 |
| `align_attachment1_all.py` | 复用首次筛查依据、补建WAV；缺少依据时生成初始候选 |
| `realign_reviewed_words.py` | 结合归档听审反馈与局部CTC复核生成采用词时间 |
| `build_final_word_features.py` | 连续处理全部显示帧，按词区间汇聚三模态并保存来源 |
| `verify_final_word_features.py` | CTC穷举、完整来源重算、时间与掩码检查 |
| `export_final_word_results.py` | 三张汇总CSV、读取说明与版本配置 |
| `feature_extractors.py`、`audio_features.py`、`media_io.py` | 冻结编码器、17维声学特征、媒体解码与真实时间 |
| `align_words.py`、`word_intervals.py` | CTC候选处理、区间交叠与来源索引 |

正式100份自包含NPZ位于 `outputs/q1_features/q1_final/features/`，字段与读取示例见该结果目录的README。

## 完整重建

按[项目运行](../../README.md#项目运行)准备环境和公开模型，放置附件一原视频与标签，安装 FFmpeg（含 ffprobe）并加入 PATH。保留 `outputs/q1_features/review_feedback/` 中的听审依据，在项目根目录依次执行：

```powershell
pixi run python src/q1_features/attachment1_inventory.py
pixi run python src/q1_features/verify_timebase_attachment1.py
pixi run python src/q1_features/extract_native_features.py
pixi run python src/q1_features/align_attachment1_all.py
pixi run python src/q1_features/realign_reviewed_words.py
pixi run python src/q1_features/build_final_word_features.py
pixi run python src/q1_features/verify_final_word_features.py
pixi run python src/q1_features/export_final_word_results.py
```
