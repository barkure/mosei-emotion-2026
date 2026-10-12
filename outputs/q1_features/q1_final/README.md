# 问题一正式结果：原生特征与词级对应

100份自包含特征位于 `features/`，每个官方样本一份NPZ及QC，序列保留实际长度。
每份同时保留原生文本词元、语音短窗、全部视频显示帧，以及按全部官方词位置组织的词级张量。

## 结果概况

- 全部100条、1932个官方空格词保留。
- 80条含采用的词时间，共1533词；其中32条所有官方词位置均通过。
- 采用词时间的位置中1533词有语音特征、1533词有视频帧显示区间、1374词有单人脸特征。
- 原始视频共处理23241帧，多人脸检出179帧，保留来源但排除其词级面部聚合。
- 20条不建立英文词时间：14条音文不一致、3条其他语言、3条静音或未听到对应语音（包含纯音乐）。
- 2条部分一致仅定位用户确认的短语；R037“Look at the record”，R043两侧短语，中间增词不归入官方词。
- 全部词时间仍是自动估计；用户听审确认的是内容及明显偏移，未提供独立精确边界标注。
- 100份NPZ合计18.00 MiB。

## 结果索引

`表2_附件1全量特征提取结果.csv`为100样本×3模态的300行表；
`word_alignment_records.csv`为全部1932词的时间、来源密度与状态；
`word_alignment_validation.json`为来源重算与CTC路径验证。

## 读取

```python
import numpy as np
with np.load("features/-egA8-b7-3M__9.npz", allow_pickle=False) as d:
    text = d["word_text_features"]   # N×768
    audio = d["word_audio_features"] # N×17
    video = d["word_video_features"] # N×52
    time_valid = d["word_time_accepted"]
    start, end = d["word_start_s"], d["word_end_s"]
    audio_valid = d["word_audio_observed"]
    face_valid = d["word_video_observed"]
```

N为全部官方空格词数量；没有词时间的词位置保留，时间为NaN、词级音视频为0、对应掩码为false。
文本词向量仍存在；“词存在”“词时间可用”“语音来源”“视频帧来源”“面部特征可用”分别保存。
数字的英文读法仅用于声学对齐，原始词和文本不改。

原生字段 `text_*`、`audio_window_*`、`vision_frame_*` 保留实际序列与独立时间。
第i词的来源由 `word_{text,audio,video}_offsets` 和 `word_{text,audio,video}_source_indices` 的CSR切片取得。
语音权重见 `word_audio_weights`，F0需再结合原生pitch掩码重加权；视频权重见 `word_video_weights`，
未检出人脸或检出多张脸的来源帧权重为0。全部被排除时，词级面部特征不可用。
视频帧显示区间为 `vision_frame_display_times_s` 到 `vision_frame_display_end_s`，原始帧号另存。
单人脸可用不等于已经验证其为说话人；回归误差或情绪判断不参与此处选择。

## 复现

```bash
pixi run python src/q1_features/extract_native_features.py
pixi run python src/q1_features/align_attachment1_all.py
pixi run python src/q1_features/realign_reviewed_words.py
pixi run python src/q1_features/build_final_word_features.py
pixi run python src/q1_features/verify_final_word_features.py
pixi run python src/q1_features/export_final_word_results.py
```

完整原视频重建步骤见附件根目录 README 的“重建与训练”部分；先准备100条解码时间轴、冻结模型及归档反馈。完整来源哈希见逐样本QC。
用户反馈及R074更正在 `outputs/q1_features/review_feedback/20260925_beb961ca30da/`。
此批听审分层选样，不用于估计无偏的总体词边界准确率。
