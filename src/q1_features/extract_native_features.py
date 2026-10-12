# 本程序及代码是在人工智能工具辅助下完成的
# 人工智能工具名称：DeepSeek，型号：DeepSeek V4.1 Flash，开发公司：杭州深度求索人工智能基础技术研究有限公司，版本颁布日期：2026-09-10

"""从全部100条原视频提取原生文本与音频特征。"""
from pathlib import Path
import argparse,json,tempfile
import numpy as np
from word_intervals import read_rows,digest
from feature_extractors import encode_text,BERT_DIR,BLENDSHAPE_NAMES_52
from audio_features import FEATURE_NAMES,window_features,mel_filterbank,hamming,SAMPLE_RATE
from media_io import decode_wav
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'outputs/q1_features'
OUT=BASE/'native_features'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--force',action='store_true')
    args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    tokenizer=model=device=None
    for row in read_rows(BASE/'attachment1_inventory.csv'):
        safe=row['stable_id'].replace('$_$','__');target=OUT/f'{safe}.npz'
        if target.exists() and not args.force:continue
        tbpath=BASE/f'attachment1_timebase/per_sample/{safe}.json'
        tb=json.loads(tbpath.read_text(encoding="utf-8"));assert tb['status']=='verified'
        if model is None:
            import torch
            from transformers import AutoModel,AutoTokenizer
            tokenizer=AutoTokenizer.from_pretrained(str(BERT_DIR),local_files_only=True)
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model=AutoModel.from_pretrained(str(BERT_DIR),local_files_only=True).eval().to(device)
        text=encode_text(row['text'],tokenizer,model,device)
        assert not text['truncated']
        video=ROOT/row['relative_path']
        with tempfile.TemporaryDirectory() as tmp:
            wav=Path(tmp)/'audio.wav';decode_wav(video,wav,SAMPLE_RATE)
            from scipy.io import wavfile
            rate,data=wavfile.read(wav);assert rate==SAMPLE_RATE and data.ndim==1
            audio=window_features(data.astype(np.float64)/32768.,mel_filterbank(),hamming())
        origin=float(tb['audio_frames']['first_frame_time'])
        output=dict(stable_id=np.array([row['stable_id']]),relative_path=np.array([row['relative_path']]),
            audio_feature_names=np.array(FEATURE_NAMES),blendshape_names=np.array(BLENDSHAPE_NAMES_52))
        for key in ['input_ids','attention_mask','token_type_ids','word_ids','last_hidden_state','cls_vector']:
            output['text_'+key]=text[key].astype(np.float16) if key in ['last_hidden_state','cls_vector'] else text[key]
        output.update(audio_window_features=audio['features'].astype(np.float32),
            audio_window_sample_start=audio['sample_start'].astype(np.int32),audio_window_sample_end=audio['sample_end'].astype(np.int32),
            audio_window_media_start_s=(origin+audio['sample_start']/SAMPLE_RATE).astype(np.float32),
            audio_window_media_end_s=(origin+audio['sample_end']/SAMPLE_RATE).astype(np.float32),
            audio_window_struct_valid=np.ones(audio['n_windows'],bool),audio_window_observed=np.ones(audio['n_windows'],bool),
            audio_window_pitch_observed=audio['pitch_observed'],audio_window_f0_at_search_boundary=audio['f0_at_boundary'])
        lineage={str(video.relative_to(ROOT)):digest(video)}
        assert not any('grid' in k for k in output)
        np.savez_compressed(target,**output)
        qc=dict(output_sha256=digest(target),source_sha256=lineage,timebase_sha256=digest(tbpath),
                code_sha256=digest(Path(__file__)),mode='raw_media_extraction')
        target.with_suffix('.json').write_text(json.dumps(qc,ensure_ascii=False,indent=2)+'\n', encoding="utf-8")
        print(safe,flush=True)

if __name__=='__main__':main()
