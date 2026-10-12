# 本程序及代码是在人工智能工具辅助下完成的
# 人工智能工具名称：DeepSeek，型号：DeepSeek V4.1 Flash，开发公司：杭州深度求索人工智能基础技术研究有限公司，版本颁布日期：2026-09-10

"""Self-contained Q1 final features: native sequences and accepted word intervals.

No fixed 0.2-second grid or artificial 50-position length. Full original display
frame sequence is processed in a single VIDEO-mode run per clip. Multiple detected
faces are recorded but excluded from word pooling because speaker identity is unknown.
"""
from __future__ import annotations
import csv,json,re
from pathlib import Path
from collections import Counter
import numpy as np
from word_intervals import digest, read_rows, write_rows, overlap_sources, pooled_audio, csr
from feature_extractors import run_face_landmarker

ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'outputs/q1_features'
OUT=BASE/'q1_final/features';RECOVERED=BASE/'attachment1_alignment/recovered'
CACHE=ROOT/'cache/q1_full_frame_features'
FEEDBACK=BASE/'review_feedback/20260925_beb961ca30da/effective_feedback.json'


def display_intervals(tb):
 v=tb['video_frames'];start=np.array(v['display_times'],float);duration=np.array(v['pkt_durations'],float)
 stop=np.r_[start[1:],float(v['coverage_end_seconds'])]
 end=np.where(np.isfinite(duration)&(duration>0),np.minimum(start+duration,stop),stop)
 assert np.all(np.diff(start)>0) and np.all(end>start)
 return start,end


def main():
 OUT.mkdir(parents=True,exist_ok=True);CACHE.mkdir(parents=True,exist_ok=True)
 summaries=[];wordrows=[]
 for k,row in enumerate(read_rows(BASE/'attachment1_inventory.csv'),1):
  sid=row['stable_id'];safe=sid.replace('$_$','__')
  native=BASE/f'native_features/{safe}.npz';alignment=RECOVERED/f'{safe}.json';timebase=BASE/f'attachment1_timebase/per_sample/{safe}.json'
  raw=dict(np.load(native,allow_pickle=False));r=json.loads(alignment.read_text(encoding="utf-8"));tb=json.loads(timebase.read_text(encoding="utf-8"))
  assert str(raw['stable_id'].item())==sid==r['stable_id']
  stamp=digest(native)+digest(alignment)+digest(timebase)+digest(Path(__file__))
  target=OUT/f'{safe}.npz';qcpath=OUT/f'{safe}_qc.json'
  if qcpath.exists() and target.exists() and json.loads(qcpath.read_text(encoding="utf-8")).get('fingerprint')==stamp:
   qc=json.loads(qcpath.read_text(encoding="utf-8"));wordrows.extend(qc['word_rows']);summaries.append(qc['summary']);print(f'{k}/100 {sid} cached',flush=True);continue
  pts,frame_end=display_intervals(tb);indices=np.arange(len(pts),dtype=np.int32)
  video=ROOT/row['relative_path'];cache=CACHE/f'{safe}.npz'
  cache_key=digest(video)+digest(timebase)+digest(ROOT/'src/q1_features/feature_extractors.py')
  if cache.exists() and str(np.load(cache)['fingerprint'])==cache_key:
   v=dict(np.load(cache));vf=v['features'];face=v['face_observed'];nf=v['n_faces'];names=v['names']
  else:
   v=run_face_landmarker(video,indices,pts,height=None)
   assert np.allclose(v['display_times'],pts,atol=5e-5,rtol=0)
   vf=v['features'].astype(np.float32);face=v['face_observed'];nf=v['n_faces'];names=np.array(v['blendshape_names'])
   np.savez_compressed(cache,fingerprint=np.array(cache_key),features=vf,face_observed=face,n_faces=nf,names=names)
  assert names.tolist()==raw['blendshape_names'].tolist()
  usable_face=face&(nf==1)
  # Keep source features even for ambiguous multiple-face frames; pooling excludes them.
  output={key:val for key,val in raw.items() if key.startswith('text_') or key.startswith('audio_window_') or key in ['stable_id','relative_path','audio_feature_names','text_cls_vector']}
  output.pop('text_clip_range_s',None)
  output.update(official_text=np.array(row['text']),label=np.array(float(row['label'])),annotation=np.array(row['annotation']),
    vision_frame_features=vf,vision_frame_display_times_s=pts,vision_frame_display_end_s=frame_end,
    vision_source_frame_indices=indices,vision_frame_struct_valid=np.ones(len(pts),bool),
    vision_frame_face_observed=face,vision_frame_n_faces=nf,vision_frame_single_face_valid=usable_face,blendshape_names=names,
    audio_clip_range_s=np.array([tb['audio_frames']['first_frame_time'],tb['audio_frames']['coverage_end_seconds']]),
    video_clip_range_s=np.array([pts[0],tb['video_frames']['coverage_end_seconds']]))
  words=list(re.finditer(r'\S+',row['text']));n=len(words);assert [m.group() for m in words]==[w['word'] for w in r['words']]
  valid=np.array([w['accepted'] for w in r['words']],bool)
  origin=float(tb['audio_frames']['first_frame_time'])
  start=np.array([w['start'] if w['accepted'] else np.nan for w in r['words']])+origin
  end=np.array([w['end'] if w['accepted'] else np.nan for w in r['words']])+origin
  text=np.zeros((n,768),np.float32);audio=np.zeros((n,17),np.float32);vision=np.zeros((n,52),np.float32)
  tm,am,vm,pm,vs=[np.zeros(n,bool) for _ in range(5)]
  token_rows=[];audio_rows=[];video_rows=[];weights=[];overlaps=[];voverlaps=[];vweights=[];records=[]
  for i,word in enumerate(r['words']):
   tok=np.flatnonzero((raw['text_word_ids']==i)&raw['text_attention_mask'].astype(bool));token_rows.append(tok)
   if len(tok):text[i]=raw['text_last_hidden_state'][tok].astype(np.float32).mean(0);tm[i]=True
   ai=np.array([],np.int32);vi=np.array([],np.int32);aw=ao=vo=vw=np.array([],float)
   if valid[i]:
    ai,ao,aw=overlap_sources(start[i],end[i],raw['audio_window_media_start_s'],raw['audio_window_media_end_s'])
    audio[i]=pooled_audio(raw['audio_window_features'],raw['audio_window_pitch_observed'],ai,aw)
    am[i]=len(ai)>0;pm[i]=raw['audio_window_pitch_observed'][ai].any()
    vi,vo,_=overlap_sources(start[i],end[i],pts,frame_end);vs[i]=len(vi)>0
    vw=vo*usable_face[vi]
    if vw.sum()>0:
     vw=vw/vw.sum();vision[i]=vw@vf[vi];vm[i]=True
   audio_rows.append(ai);video_rows.append(vi);weights.extend(aw);overlaps.extend(ao);voverlaps.extend(vo);vweights.extend(vw)
   records.append(dict(stable_id=sid,word_index=i,word=word['word'],start_s=float(start[i]) if valid[i] else '',
     end_s=float(end[i]) if valid[i] else '',time_accepted=bool(valid[i]),audio_observed=bool(am[i]),
     video_source_observed=bool(vs[i]),face_observed=bool(vm[i]),source_audio_windows=len(ai),
     source_video_frames=len(vi),single_face_frames=int(usable_face[vi].sum()),reason=word['reason']))
  output.update(official_words=np.array([m.group() for m in words]),word_char_start=np.array([m.start() for m in words]),
   word_char_end=np.array([m.end() for m in words]),word_struct_valid=np.ones(n,bool),word_time_accepted=valid,
   word_start_s=start,word_end_s=end,word_text_features=text,word_audio_features=audio,word_video_features=vision,
   word_text_observed=tm,word_audio_observed=am,word_video_observed=vm,word_video_source_observed=vs,
   word_pitch_observed=pm,word_category=np.array(r['category']),word_quality_reason=np.array([w['reason'] for w in r['words']]),
   word_normalized_target=np.array([w['normalized_target'] for w in r['words']]),
   word_content_human_reviewed=np.array([w['human_content_reviewed'] for w in r['words']]),
   word_boundary_human_annotated=np.zeros(n,bool),
   word_audio_weights=np.array(weights),word_audio_overlap_s=np.array(overlaps),
   word_video_weights=np.array(vweights),word_video_overlap_s=np.array(voverlaps))
  for name,rs in [('text',token_rows),('audio',audio_rows),('video',video_rows)]:
   output[f'word_{name}_offsets'],output[f'word_{name}_source_indices']=csr(rs)
  for key in ['word_text_features','word_audio_features','word_video_features','vision_frame_features']:assert np.isfinite(output[key]).all()
  assert np.all(~vm|vs) and np.all(~vs|valid)
  np.savez_compressed(target,**output)
  summary=dict(stable_id=sid,category=r['category'],official_words=n,accepted_words=int(valid.sum()),
   audio_words=int(am.sum()),video_source_words=int(vs.sum()),face_words=int(vm.sum()),
   full_video_frames=len(pts),frames_with_face=int(face.sum()),multi_face_frames=int((nf>1).sum()),
   native_audio_windows=len(raw['audio_window_features']),text_tokens=len(raw['text_word_ids']),
   audio_end_s=tb['audio_frames']['coverage_end_seconds'],video_end_s=tb['video_frames']['coverage_end_seconds'],file_bytes=target.stat().st_size)
  qc=dict(fingerprint=stamp,summary=summary,word_rows=records,output_sha256=digest(target),
   source_sha256={str(p.relative_to(ROOT)):digest(p) for p in [native,alignment,timebase,FEEDBACK]},
   video_policy='All original display frames in increasing PTS, one VIDEO-mode session; blendshape from returned face 0 retained, n_faces != 1 excluded from word pooling. This is not active-speaker verification.',
   video_interval_policy='[PTS, min(PTS+positive_packet_duration,next_PTS)); missing duration uses next_PTS; last uses decoded coverage end.',
   word_time_policy='Acoustic-local CTC time plus decoded audio origin; no independent human boundary annotation.')
  qcpath.write_text(json.dumps(qc,ensure_ascii=False,indent=2)+'\n', encoding="utf-8");summaries.append(summary);wordrows.extend(records)
  print(f'{k}/100 {sid}: {valid.sum()} words, {len(pts)} frames',flush=True)
 write_rows(OUT.parent/'word_alignment_summary.csv',summaries);write_rows(OUT.parent/'word_alignment_records.csv',wordrows)
 summary=dict(n_samples=100,n_official_words=sum(r['official_words'] for r in summaries),
  accepted_words=sum(r['accepted_words'] for r in summaries),samples_with_word_times=sum(r['accepted_words']>0 for r in summaries),
  fully_accepted_samples=sum(r['accepted_words']==r['official_words'] for r in summaries),
  audio_words=sum(r['audio_words'] for r in summaries),video_source_words=sum(r['video_source_words'] for r in summaries),
  face_words=sum(r['face_words'] for r in summaries),video_frames=sum(r['full_video_frames'] for r in summaries),
  multi_face_frames=sum(r['multi_face_frames'] for r in summaries),feature_bytes=sum(r['file_bytes'] for r in summaries),
  category_counts=dict(Counter(r['category'] for r in summaries)),fixed_time_grid=False,fixed_sequence_length=False,
  code_sha256=digest(Path(__file__)))
 (OUT.parent/'word_alignment_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n', encoding="utf-8");print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
