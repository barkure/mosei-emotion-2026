# 本程序及代码是在人工智能工具辅助下完成的
# 人工智能工具名称：DeepSeek，型号：DeepSeek V4.1 Flash，开发公司：杭州深度求索人工智能基础技术研究有限公司，版本颁布日期：2026-09-10

"""Q1: content-reviewed, locally checked CTC word alignment.

Official text is immutable. Human content checks do not annotate boundaries.
CTC posteriors use the existing frozen English model. Native features are separate.
"""
from __future__ import annotations
import csv, difflib, hashlib, json, re
from pathlib import Path
from collections import Counter
import numpy as np
from scipy.io import wavfile
from align_words import SNAPSHOT, MODEL_ID, MODEL_REVISION, load_vocab, min_required_frames, detect_silence

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'outputs/q1_features'
OUT=BASE/'attachment1_alignment/recovered'
FEEDBACK=BASE/'review_feedback/20260925_beb961ca30da/effective_feedback.json'
STEP=.02
CONFIDENCE=.5
STABILITY=.10
POLICY='reviewed-local-ctc-v1'
NUMBER_FORMS={'2008':['two thousand eight','two thousand and eight'],
 '1500':['fifteen hundred','one thousand five hundred'],
 '100000':['one hundred thousand','a hundred thousand'],
 '20000':['twenty thousand'],'50':['fifty'],'10th':['tenth'],
 '1989':['nineteen eighty nine','one thousand nine hundred eighty nine']}
PARTIAL_PHRASES={
 '-HwX2H8Z4hY$_$2':['Look at the record'],
 '-aqamKhZ1Ec$_$0':['tribal warfare in Kenya','following a disputed election'],
}


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def write_rows(p,rows):
 with p.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def ctc_path(lp,targets):
 """Vectorized Viterbi; return first AND last emitting frames, excluding blanks."""
 T,V=lp.shape
 if not targets or T<min_required_frames(targets):raise ValueError('Unreachable CTC target')
 labels=np.zeros(2*len(targets)+1,dtype=np.int64);labels[1::2]=targets
 E=len(labels);neg=-np.inf
 previous=np.full(E,neg);previous[:2]=lp[0,labels[:2]]
 back=np.zeros((T,E),np.int8)
 skip=np.zeros(E,bool);skip[2:]=(labels[2:]!=0)&(labels[2:]!=labels[:-2])
 for t in range(1,T):
  advance=np.r_[neg,previous[:-1]]
  jump=np.r_[neg,neg,previous[:-2]];jump[~skip]=neg
  options=np.stack([previous,advance,jump])
  move=options.argmax(axis=0);back[t]=move
  previous=options[move,np.arange(E)]+lp[t,labels]
 state=E-1 if previous[-1]>=previous[-2] else E-2
 score=previous[state]
 if not np.isfinite(score):raise ValueError('No finite CTC path')
 path=np.empty(T,np.int32)
 for t in range(T-1,-1,-1):path[t]=state;state-=int(back[t,state])
 starts=[];ends=[];conf=[]
 for j,token in enumerate(targets):
  frames=np.flatnonzero(path==2*j+1)
  if not len(frames):raise ValueError('Missing emitting state')
  starts.append(int(frames[0]));ends.append(int(frames[-1]+1))
  conf.append(float(np.exp(lp[frames,token].mean())))
 return np.array(starts),np.array(ends),np.array(conf),float(score)


def normalize(word,greedy):
 stripped=word.lower().strip('.,:;!?[]()"“”^').replace(',','')
 if any(c.isdigit() for c in word):
  forms=NUMBER_FORMS.get(stripped)
  if not forms:return '',dict(kind='unsupported_number',original=word)
  # Prefer an exact reading present in diagnostic ASR; otherwise use declared first form.
  picked=next((s for s in forms if s.upper() in greedy.upper()),forms[0])
  return picked.upper(),dict(kind='number_reading',original=word,chosen=picked,alternatives=forms)
 normalized=re.sub("[^A-Z' ]",'',word.upper())
 return normalized,dict(kind='letter_normalization',original=word,chosen=normalized,
                       dropped=[c for c in word.upper() if not (c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ' ")])


def targets_for(norms,vocab):
 tokens=[];positions=[]
 for word in norms:
  if tokens:tokens.append(vocab['|'])
  indexes=[]
  for c in word:
   token=vocab['|'] if c==' ' else vocab[c]
   indexes.append(len(tokens));tokens.append(token)
  positions.append(indexes)
 return tokens,positions


def align(lp,norms,vocab,offset=0):
 targets,positions=targets_for(norms,vocab)
 starts,ends,confidence,score=ctc_path(lp,targets)
 rows=[]
 for pos in positions:
  letters=[j for j in pos if targets[j]!=vocab['|']]
  if not letters:rows.append(None);continue
  rows.append(dict(start=round((offset+starts[letters[0]])*STEP,6),
                   end=round((offset+ends[letters[-1]])*STEP,6),
                   confidence=float(confidence[letters].mean()),
                   confidence_min=float(confidence[letters].min())))
 return rows,score


def greedy_decode(lp,vocab):
 inverse={v:k for k,v in vocab.items()};ids=lp.argmax(1);chars=[];times=[];last=-1
 for i,k in enumerate(ids):
  if k and k!=last:chars.append(inverse[int(k)]);times.append(i)
  last=k
 text=''.join(chars).replace('|',' ')
 return text,np.array(times)


def locate_phrase(phrase,greedy,frames):
 """Locate a supported phrase using a contiguous ASR character window.

 Used only after explicit human confirmation of that phrase; no equal-time split.
 """
 query=phrase.upper();best=(-1,None)
 words=list(re.finditer(r'\S+',greedy));n=len(query.split())
 for a in range(len(words)):
  for size in range(max(2,n-1),n+3):
   b=a+size
   if b>len(words):continue
   l,r=words[a].start(),words[b-1].end()
   score=difflib.SequenceMatcher(None,query,greedy[l:r],autojunk=False).ratio()
   if score>best[0]:best=(score,(l,r))
 if best[1] is None or best[0]<.6:return None
 l,r=best[1]
 return max(0,int(frames[l])-8),int(frames[r-1])+10,best[0],greedy[l:r]


def emissions(wav,model_bundle):
 extractor,model,torch=model_bundle
 cache=ROOT/'cache/q1_reviewed_ctc';cache.mkdir(parents=True, exist_ok=True)
 p=cache/(wav.stem+'.npz');fingerprint=sha(wav)+MODEL_REVISION
 if p.exists():
  z=np.load(p)
  if str(z['fingerprint'])==fingerprint:return z['log_probs']
 rate,x=wavfile.read(wav);assert rate==16000 and x.ndim==1
 inputs=extractor(x.astype(np.float32)/32768.,sampling_rate=rate,return_tensors='pt')
 with torch.inference_mode():lp=model(inputs.input_values.to(model.device)).logits[0].log_softmax(-1).cpu().numpy()
 np.savez_compressed(p,log_probs=lp,fingerprint=np.array(fingerprint))
 return lp


def main():
 import torch
 from transformers import Wav2Vec2FeatureExtractor,Wav2Vec2ForCTC
 OUT.mkdir(parents=True,exist_ok=True);vocab=load_vocab()
 feedback=json.loads(FEEDBACK.read_text(encoding="utf-8"));by={}
 for r in feedback['items']:by.setdefault(r['stable_id'],[]).append(r)
 extractor=Wav2Vec2FeatureExtractor.from_pretrained(str(SNAPSHOT))
 model=Wav2Vec2ForCTC.from_pretrained(str(SNAPSHOT)).to('cuda' if torch.cuda.is_available() else 'cpu').eval()
 torch.set_num_threads(4)
 with (BASE/'attachment1_inventory.csv').open(encoding='utf-8', newline='') as f:samples=list(csv.DictReader(f))
 summaries=[];allwords=[]
 for k,s in enumerate(samples,1):
  sid=s['stable_id'];safe=sid.replace('$_$','__');oldpath=BASE/f'attachment1_alignment/all_candidates/{safe}.json'
  old=json.loads(oldpath.read_text(encoding="utf-8"));reviews=by.get(sid,[]);whole=next((r for r in reviews if r['review_type']=='whole_clip'),None)
  old_pass=old['alignment_status']=='candidate_for_manual_spot_check'
  if sid in PARTIAL_PHRASES:category='human_partial_content'
  elif whole and whole['verdict']!='内容一致，未发现明显问题':category={'实际内容与文本不一致':'content_mismatch','其他语言':'other_language','静音或未听到语音':'no_audible_speech'}[whole['verdict']]
  elif whole:category='human_whole_content_confirmed'
  elif not old_pass:category='human_flagged_words_confirmed'
  else:category='previous_auto_accepted'
  words=s['text'].split();wav=BASE/f'attachment1_alignment/audio/{safe}.wav'
  fingerprint=sha(oldpath)+sha(FEEDBACK)+sha(wav)+sha(Path(__file__))
  dest=OUT/f'{safe}.json'
  if dest.exists() and json.loads(dest.read_text(encoding="utf-8")).get('fingerprint')==fingerprint:
   result=json.loads(dest.read_text(encoding="utf-8"))
  else:
   enabled=category not in ['content_mismatch','other_language','no_audible_speech']
   norm_pairs=[normalize(w,old['diagnostic_asr']['greedy_text']) for w in words]
   norms=[x[0] for x in norm_pairs];normmeta=[x[1] for x in norm_pairs]
   results=[dict(word_index=i,word=w,normalized_target=norms[i],normalization=normmeta[i],
                 candidate_start=None,candidate_end=None,start=None,end=None,confidence=None,
                 local_confidence=None,boundary_shift_s=None,accepted=False,
                 reason=category if not enabled else 'unverified',human_content_reviewed=bool(whole) or any(r['word_index']==i for r in reviews)) for i,w in enumerate(words)]
   blocks=[];silence=[];greedy=''
   if enabled:
    lp=emissions(wav,(extractor,model,torch));greedy,gframes=greedy_decode(lp,vocab)
    silence=detect_silence(wav)['segments']
    if category=='human_partial_content':
     official_norm=' '.join(norms)
     for phrase in PARTIAL_PHRASES[sid]:
      normphrase=phrase.upper();ids=None
      for i in range(len(words)):
       for j in range(i+1,len(words)+1):
        if ' '.join(norms[i:j])==normphrase:ids=list(range(i,j))
      located=locate_phrase(phrase,greedy,gframes)
      if ids is not None and located:
       l,r,score,matched=located;blocks.append(dict(indices=ids,left=l,right=min(r,len(lp)),phrase_locator_similarity=score,matched_asr=matched))
     for rr in results:rr['reason']='outside_human_confirmed_phrase'
    else:
     blocks=[dict(indices=list(range(len(words))),left=0,right=len(lp))]
    for block in blocks:
     ids=block['indices'];left,right=block['left'],block['right'];ns=[norms[i] for i in ids]
     try:global_rows,_=align(lp[left:right],ns,vocab,left)
     except ValueError:
      for i in ids:results[i]['reason']='ctc_path_unreachable'
      continue
     for j,i in enumerate(ids):
      rr=results[i];g=global_rows[j]
      if not g:rr['reason']='no_supported_target_characters';continue
      rr.update(candidate_start=g['start'],candidate_end=g['end'],confidence=g['confidence'],confidence_min=g['confidence_min'])
      # Overlapping contexts provide a local boundary check with fixed 100-ms tolerance.
      a=max(0,j-3);b=min(len(ids),j+4)
      nonempty=[v for v in global_rows[a:b] if v]
      lo=max(left,round(nonempty[0]['start']/STEP)-8)
      hi=min(right,round(nonempty[-1]['end']/STEP)+8)
      try:
       local,_=align(lp[lo:hi],[norms[x] for x in ids[a:b]],vocab,lo);v=local[j-a]
       if not v:raise ValueError('No local word')
      except ValueError:rr['reason']='local_path_unreachable';continue
      shift=max(abs(g['start']-v['start']),abs(g['end']-v['end']))
      rr.update(local_start=v['start'],local_end=v['end'],local_confidence=v['confidence'],boundary_shift_s=round(shift,6))
      entirely_silent=any(v['start']>=z['start'] and v['end']<=(z['end'] if z['end'] is not None else len(lp)*STEP) for z in silence)
      duration=v['end']-v['start'];max_duration=max(1.,.2*len(norms[i].replace(' ','')))
      if min(g['confidence'],v['confidence'])<CONFIDENCE:reason='low_character_confidence'
      elif shift>STABILITY+1e-8:reason='unstable_local_boundary'
      elif entirely_silent:reason='inside_detected_silence'
      elif duration>max_duration:reason='excessive_word_span'
      elif not 0<=v['start']<v['end']<=len(lp)*STEP+1e-8:reason='invalid_time'
      else:reason='accepted_automatic_with_content_evidence';rr.update(accepted=True,start=v['start'],end=v['end'])
      rr['reason']=reason
    # Differently cropped contexts must not create overlapping accepted words.
    previous=None
    for rr in results:
     if not rr['accepted']:continue
     if previous and rr['start']<previous['end']-1e-8:
      for x in [previous,rr]:x.update(accepted=False,start=None,end=None,reason='overlap_after_local_alignment')
      previous=None
     else:previous=rr
   result=dict(stable_id=sid,official_text=s['text'],category=category,words=results,
               blocks=blocks,greedy_text=greedy,silence=silence,fingerprint=fingerprint,
               evidence_review_ids=[r['review_id'] for r in reviews],source_candidate_sha256=sha(oldpath),
               feedback_sha256=sha(FEEDBACK),audio_sha256=sha(wav),model=MODEL_ID,revision=MODEL_REVISION,
               policy=dict(name=POLICY,character_confidence_min=CONFIDENCE,local_boundary_tolerance_s=STABILITY,
                           local_context_words_each_side=3,context_padding_s=.16,
                           maximum_word_span='max(1.0, 0.2 * normalized_letter_count) seconds',
                           note='Heuristic acceptance; content reviews are not independent word-boundary labels.'))
   dest.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n', encoding="utf-8")
  n=sum(w['accepted'] for w in result['words'])
  summaries.append(dict(stable_id=sid,category=category,n_words=len(words),accepted_words=n,
                        all_words_accepted=n==len(words),original_accepted_words=sum(w.get('usable_for_feature_alignment',False) for w in old['words'])))
  allwords.extend(dict(stable_id=sid,category=category,**{key:w.get(key) for key in ['word_index','word','normalized_target','candidate_start','candidate_end','start','end','confidence','local_confidence','boundary_shift_s','accepted','reason','human_content_reviewed']}) for w in result['words'])
  print(f'{k}/100 {sid} {category}: {n}/{len(words)}',flush=True)
 write_rows(OUT/'summary.csv',summaries);write_rows(OUT/'words.csv',allwords)
 summary=dict(samples=100,official_words=sum(s['n_words'] for s in summaries),accepted_words=sum(s['accepted_words'] for s in summaries),
              samples_with_accepted_words=sum(s['accepted_words']>0 for s in summaries),fully_accepted_samples=sum(s['all_words_accepted'] for s in summaries),
              categories=dict(Counter(s['category'] for s in summaries)),word_reasons=dict(Counter(w['reason'] for w in allwords)),
              model=MODEL_ID,revision=MODEL_REVISION,feedback_sha256=sha(FEEDBACK),code_sha256=sha(Path(__file__)))
 (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n', encoding="utf-8");print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
