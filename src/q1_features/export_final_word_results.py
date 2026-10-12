# 本程序及代码是在人工智能工具辅助下完成的
# 人工智能工具名称：DeepSeek，型号：DeepSeek V4.1 Flash，开发公司：杭州深度求索人工智能基础技术研究有限公司，版本颁布日期：2026-09-10

"""Export canonical Q1 tables for native sequences and reviewed word alignment."""
from pathlib import Path
import json,hashlib,subprocess
import numpy as np
from importlib.metadata import version
from word_intervals import read_rows,write_rows
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'outputs/q1_features';DEST=BASE/'q1_final'
CATEGORY={
 'previous_auto_accepted':'自动筛查样本，逐词复核时间',
 'human_flagged_words_confirmed':'可疑词内容经听审确认',
 'human_whole_content_confirmed':'整段内容经听审确认',
 'human_partial_content':'仅处理听审确认的短语',
 'content_mismatch':'音文内容不一致',
 'other_language':'其他语言',
 'no_audible_speech':'未听到对应语音（含静音或纯音乐）'}


def main():
 summary=json.loads((DEST/'word_alignment_summary.json').read_text(encoding="utf-8"))
 rows=read_rows(DEST/'word_alignment_summary.csv');assert len(rows)==100
 configs=[
  dict(项目='正式组织方式',配置='原生序列＋具有证据的词区间对应',说明='按全部官方词位置组织三模态张量，保留实际长度'),
  dict(项目='文本',配置='冻结bert-base-uncased；同一官方词的子词均值',说明='词元及每词768维；官方原文及词位置完整保留'),
  dict(项目='语音',配置='16 kHz；25 ms窗／10 ms步长；17维',说明='原生窗与词区间交叠加权；F0仅对可测窗重新归一化'),
  dict(项目='视频',配置='全部原始显示帧；宽640；MediaPipe VIDEO；52维',说明='每片段一次按时间连续推理；保留检测数，多人脸帧不参与词特征聚合'),
  dict(项目='帧时间区间',配置='PTS与正的帧持续时间，终点不超过下一PTS',说明='词区间与帧显示区间交叠加权；源帧与面部特征掩码分开'),
  dict(项目='词时间',配置='冻结wav2vec2-base-960h；CTC路径＋局部上下文检查',说明='字符均值置信度≥0.5、局部边界差≤0.10秒；检查静音、跨度、顺序与范围'),
  dict(项目='人工作用',配置='94项内容与明显偏移听审；原始反馈与更正分开保存',说明='用于确定音文关系和局部处理范围，未提供独立精确边界标签'),
  dict(项目='异常样本',配置='14条音文不一致、3条其他语言、3条无对应语音',说明='保留原始文本和音视频特征，词时间不可用'),
 ]
 write_rows(DEST/'表1_特征提取与对齐总体配置.csv',configs)
 table=[]
 for r in rows:
  n=int(r['official_words']);a=int(r['accepted_words']);sid=r['stable_id'];p=f"outputs/q1_features/q1_final/features/{sid.replace('$_$','__')}.npz"
  with np.load(ROOT/p,allow_pickle=False) as features:
   durations={modality:float(np.diff(features[f'{key}_clip_range_s'])[0])
              for modality,key in [('语音','audio'),('视频','video')]}
  for modality,dim,source_len,origin_end,available in [
   ('文本',768,r['text_tokens'],'所属样本范围；词时间见NPZ',n),
   ('语音',17,r['native_audio_windows'],r['audio_end_s'],r['audio_words']),
   ('视频',52,r['full_video_frames'],r['video_end_s'],r['face_words'])]:
   table.append(dict(样本片段编号=sid,模态=modality,原生位置数=source_len,特征维数=dim,
    解码覆盖时长_s=durations.get(modality,'文本无独立物理时长'),原生覆盖终点_s=origin_end,
    官方词数=n,采用词时间数=a,词级有效特征数=available,词级张量形状=f'{n}×{dim}',
    对齐粒度='可用词区间' if a else '无可用词时间；保留原生序列',音文状态=CATEGORY[r['category']],特征文件=p))
 write_rows(DEST/'表2_附件1全量特征提取结果.csv',table)
 cats=summary['category_counts']
 write_rows(DEST/'表3_质量标记定义与异常统计.csv',[dict(状态码=k,说明=CATEGORY[k],样本数=v) for k,v in cats.items()])
 text=f'''# 问题一正式结果：原生特征与词级对应

100份自包含特征位于 `features/`，每个官方样本一份NPZ及QC，序列保留实际长度。
每份同时保留原生文本词元、语音短窗、全部视频显示帧，以及按全部官方词位置组织的词级张量。

## 结果概况

- 全部100条、1932个官方空格词保留。
- {summary['samples_with_word_times']}条含采用的词时间，共{summary['accepted_words']}词；其中{summary['fully_accepted_samples']}条所有官方词位置均通过。
- 采用词时间的位置中{summary['audio_words']}词有语音特征、{summary['video_source_words']}词有视频帧显示区间、{summary['face_words']}词有单人脸特征。
- 原始视频共处理{summary['video_frames']}帧，多人脸检出{summary['multi_face_frames']}帧，保留来源但排除其词级面部聚合。
- 20条不建立英文词时间：14条音文不一致、3条其他语言、3条静音或未听到对应语音（包含纯音乐）。
- 2条部分一致仅定位用户确认的短语；R037“Look at the record”，R043两侧短语，中间增词不归入官方词。
- 全部词时间仍是自动估计；用户听审确认的是内容及明显偏移，未提供独立精确边界标注。
- 100份NPZ合计{summary['feature_bytes']/1024**2:.2f} MiB。

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
第i词的来源由 `word_{{text,audio,video}}_offsets` 和 `word_{{text,audio,video}}_source_indices` 的CSR切片取得。
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
'''
 (DEST/'README.md').write_text(text, encoding="utf-8")
 sources=['src/q1_features/realign_reviewed_words.py','src/q1_features/build_final_word_features.py',
          'src/q1_features/verify_final_word_features.py','src/q1_features/export_final_word_results.py',
          'src/q1_features/word_intervals.py','src/q1_features/extract_native_features.py','src/q1_features/feature_extractors.py',
          'src/q1_features/audio_features.py','src/q1_features/media_io.py',
          'src/q1_features/align_words.py','models/mediapipe/face_landmarker.task',
          'outputs/q1_features/review_feedback/20260925_beb961ca30da/effective_feedback.json']
 protocol=dict(runtime_versions={p:version(p) for p in ['numpy','scipy','torch','transformers','mediapipe']},
               ffmpeg_version=subprocess.check_output(['ffmpeg','-version'],text=True).splitlines()[0],
               source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources},
               audio=dict(sample_rate=16000,window_samples=400,hop_samples=160,feature_dimension=17),
               video=dict(mode='VIDEO',all_display_frames=True,width=640,num_faces=2,
                          min_face_detection_confidence=.5,min_face_presence_confidence=.5,min_tracking_confidence=.5,
                          word_pooling='display interval overlap, only n_faces == 1'),
               word_time=dict(model="facebook/wav2vec2-base-960h",revision="22aad52d435eb6dbaf354bdad9b0da84ce7d6156",ctc_stride_s=.02,confidence_min=.5,local_boundary_difference_max_s=.1,
                              independent_human_boundary_labels=False),
               text=dict(model='google-bert/bert-base-uncased',revision='86b5e0934494bd15c9632b12f734a8a67f723594',
                         hidden_dimension=768,frozen=True))
 (DEST/'alignment_protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n', encoding="utf-8")

 print('Exported current Q1 tables: 100 self-contained feature files, 300 modality rows.')

if __name__=='__main__':main()
