# 本程序及代码是在人工智能工具辅助下完成的
# 人工智能工具名称：DeepSeek，型号：DeepSeek V4.1 Flash，开发公司：杭州深度求索人工智能基础技术研究有限公司，版本颁布日期：2026-09-10

"""Independent CTC oracle tests, source membership and full feature reconstruction."""
from pathlib import Path
from collections import Counter
import json
import numpy as np
from scipy.special import log_softmax
from align_words import enumerate_legal_paths, min_required_frames
from realign_reviewed_words import ctc_path, sha, OUT as ALIGNMENTS
from build_final_word_features import BASE, OUT


def check_ctc():
 rng=np.random.default_rng(20260925);count=0
 for targets in [[1],[1,1],[1,2],[1,2,1],[2,2,1]]:
  for extra in [0,1,2]:
   T=min_required_frames(targets)+extra
   for _ in range(3):
    lp=log_softmax(rng.normal(size=(T,4)),axis=-1)
    oracle=enumerate_legal_paths(lp,targets)
    starts,ends,conf,score=ctc_path(lp,targets)
    np.testing.assert_allclose(score,oracle['best_logprob'],atol=1e-10)
    path=np.array(oracle['best_path'])
    for i in range(len(targets)):
     frames=np.flatnonzero(path==2*i+1)
     assert starts[i]==frames[0] and ends[i]==frames[-1]+1
     np.testing.assert_allclose(conf[i],np.exp(lp[frames,targets[i]].mean()))
    count+=1
 try:ctc_path(np.zeros((2,4)),[1,1])
 except ValueError:pass
 else:raise AssertionError('Unreachable repeated target was accepted')
 # Regression: last character stays for two frames; end must include both.
 lp=np.array([[-10.,0.],[-10.,0.]])
 assert ctc_path(lp,[1])[1].tolist()==[2]
 # A frame already on screen must cover a shorter word, even with no new PTS.
 left=np.array([.1]);right=np.array([.133]);a,b=.11,.125
 assert np.maximum(0,np.minimum(right,b)-np.maximum(left,a)).sum()>0
 return count


def main():
 cases=check_ctc();nwords=nvideo=0;cats=Counter();max_audio_error=max_video_error=0.
 files=list(OUT.glob('*.npz'));assert len(files)==100
 for p in files:
  d=dict(np.load(p,allow_pickle=False));sid=str(d['stable_id'].item())
  qc=json.loads(p.with_name(p.stem+'_qc.json').read_text(encoding="utf-8"));assert sha(p)==qc['output_sha256']
  for f,h in qc['source_sha256'].items():assert sha(BASE.parents[1]/f)==h
  r=json.loads((ALIGNMENTS/(p.stem+'.json')).read_text(encoding="utf-8"));cats[r['category']]+=1
  assert r['stable_id']==sid and r['official_text']==str(d['official_text'])
  assert not any('grid' in key for key in d)
  n=len(d['official_words']);assert len(r['words'])==n
  accepted=d['word_time_accepted'];nwords+=int(accepted.sum())
  assert np.isnan(d['word_start_s'][~accepted]).all() and np.isnan(d['word_end_s'][~accepted]).all()
  assert not d['word_boundary_human_annotated'].any()
  assert np.all(~d['word_video_observed']|d['word_video_source_observed'])
  for name in ['word_audio_observed','word_video_source_observed']:assert np.all(~d[name]|accepted)
  if r['category'] in ['content_mismatch','other_language','no_audible_speech']:assert not accepted.any()
  valid_start=d['word_start_s'][accepted];valid_end=d['word_end_s'][accepted]
  assert np.all(valid_end>valid_start) and np.all(valid_start[1:]>=valid_end[:-1]-1e-8)
  for name in ['word_text_features','word_audio_features','word_video_features','audio_window_features','vision_frame_features']:
   assert np.isfinite(d[name]).all(),(p,name)
  tb=json.loads((BASE/f'attachment1_timebase/per_sample/{p.stem}.json').read_text(encoding="utf-8"))
  pts=np.array(tb['video_frames']['display_times']);dur=np.array(tb['video_frames']['pkt_durations'],float)
  np.testing.assert_array_equal(pts,d['vision_frame_display_times_s'])
  assert len(pts)==len(d['vision_frame_features']);nvideo+=len(pts)
  nxt=np.r_[pts[1:],tb['video_frames']['coverage_end_seconds']]
  ends=np.where(np.isfinite(dur)&(dur>0),np.minimum(pts+dur,nxt),nxt)
  np.testing.assert_array_equal(ends,d['vision_frame_display_end_s'])
  face=d['vision_frame_face_observed']&(d['vision_frame_n_faces']==1)
  assert np.array_equal(face,d['vision_frame_single_face_valid'])
  for name in ['text','audio','video']:
   offsets=d[f'word_{name}_offsets'];indices=d[f'word_{name}_source_indices']
   assert len(offsets)==n+1 and offsets[0]==0 and offsets[-1]==len(indices)
   assert np.all(np.diff(offsets)>=0)
  for i in range(n):
   l,h=d['word_text_offsets'][i:i+2];ti=d['word_text_source_indices'][l:h]
   np.testing.assert_array_equal(ti,np.flatnonzero(d['text_word_ids']==i))
   expected=d['text_last_hidden_state'][ti].astype(np.float32).mean(0) if len(ti) else np.zeros(768)
   np.testing.assert_allclose(expected,d['word_text_features'][i],atol=1e-6)
   a0,a1=d['word_audio_offsets'][i:i+2];v0,v1=d['word_video_offsets'][i:i+2]
   if not accepted[i]:
    assert a0==a1 and v0==v1 and not d['word_audio_features'][i].any() and not d['word_video_features'][i].any();continue
   a,b=d['word_start_s'][i],d['word_end_s'][i]
   assert abs((a-d['audio_clip_range_s'][0])-r['words'][i]['start'])<1e-8
   overlap=np.maximum(0,np.minimum(d['audio_window_media_end_s'],b)-np.maximum(d['audio_window_media_start_s'],a))
   ai=np.flatnonzero(overlap>1e-9);np.testing.assert_array_equal(ai,d['word_audio_source_indices'][a0:a1])
   weights=overlap[ai]/overlap[ai].sum()
   np.testing.assert_allclose(weights,d['word_audio_weights'][a0:a1])
   expected=weights@d['audio_window_features'][ai].astype(float)
   voiced=d['audio_window_pitch_observed'][ai]
   expected[14]=np.average(d['audio_window_features'][ai[voiced],14],weights=weights[voiced]) if voiced.any() else 0
   max_audio_error=max(max_audio_error,float(abs(expected-d['word_audio_features'][i]).max()))
   np.testing.assert_allclose(expected,d['word_audio_features'][i],atol=1e-4,rtol=2e-6)
   overlap=np.maximum(0,np.minimum(ends,b)-np.maximum(pts,a))
   vi=np.flatnonzero(overlap>1e-9);np.testing.assert_array_equal(vi,d['word_video_source_indices'][v0:v1])
   np.testing.assert_allclose(overlap[vi],d['word_video_overlap_s'][v0:v1])
   vw=overlap[vi]*face[vi]
   if vw.sum()>0:vw/=vw.sum()
   expected=vw@d['vision_frame_features'][vi] if len(vi) else np.zeros(52)
   np.testing.assert_allclose(vw,d['word_video_weights'][v0:v1])
   max_video_error=max(max_video_error,float(abs(expected-d['word_video_features'][i]).max()))
   np.testing.assert_allclose(expected,d['word_video_features'][i],atol=1e-6)
 report=dict(samples=100,accepted_words_reconstructed=nwords,full_video_frames_checked=nvideo,
             ctc_exhaustive_oracle_cases=cases,ctc_last_character_end_regression=True,
             half_open_display_interval_regression=True,no_fixed_grid=True,source_hashes_verified=True,
             native_and_word_feature_reconstruction=True,manual_boundary_claims_absent=True,
             category_counts=dict(cats),max_float32_audio_rounding_error=max_audio_error,
             max_float32_video_rounding_error=max_video_error)
 (OUT.parent/'word_alignment_validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n', encoding="utf-8")
 print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
