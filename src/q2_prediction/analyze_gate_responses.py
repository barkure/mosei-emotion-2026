# 本程序及代码是在人工智能工具辅助下完成的
# 人工智能工具名称：ChatGPT，型号：GPT-6 Astra，开发公司：OpenAI，版本颁布日期：2026-09-04
"""Observe fixed MoE gates on attachment 2 validation; no fitting or selection."""
from pathlib import Path
import argparse
import hashlib
import json
import pickle
import sys
import numpy as np
import torch
from transformers import AutoModel
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'src'), str(Path(__file__).resolve().parent)]
from common.prep import BERT, DATA, make_split, slice_batch, apply_missing
from experiment_q2_mask_fusion import Specialist, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    folder = ROOT/'outputs/q2_prediction/q2_systematic'
    seeds = [2026, 2027, 2028]
    saves = [torch.load(folder/f'gate_recon_seed{s}.pt', map_location='cpu', weights_only=False) for s in seeds]
    with DATA.open('rb') as f:
        fields = pickle.load(f)['valid']
    split = make_split(fields, saves[0]['normalizers'], torch.device(args.device))
    bert = AutoModel.from_pretrained(str(BERT), local_files_only=True).to(args.device).eval()
    bert.requires_grad_(False)
    models = []
    for saved in saves:
        assert saved['mode'] == 'gate_recon' and saved['data_version'] == 'aligned_50'
        model = Specialist('gate_recon').to(args.device).eval()
        model.load_state_dict(saved['state_dict'])
        models.append(model)
    result = dict(split='attachment2_valid', n=len(fields['id']), seeds=seeds,
                  sample_ids=list(fields['id']), aggregation='per-sample arithmetic mean of three gates',
                  checkpoint_sha256=[hashlib.sha256((folder/f'gate_recon_seed{s}.pt').read_bytes()).hexdigest() for s in seeds], scenarios={})
    for mod in [None, 'text', 'audio', 'vision']:
        rows, predictions, strengths = [], [], []
        with torch.inference_mode():
            for start in range(0, len(fields['id']), 64):
                ids = np.arange(start, min(start+64, len(fields['id'])))
                batch = slice_batch(split, ids)
                if mod:
                    apply_missing(batch, mod, .4, 'middle')
                gates, logits, raw = [], [], []
                for model in models:
                    captured = []
                    hook = model.gate.register_forward_hook(lambda module, inputs, output: captured.append(output))
                    try:
                        out, strength, aux = model(batch, bert)
                    finally:
                        hook.remove()
                    g = torch.sigmoid(captured[0]).squeeze(-1)
                    g = torch.where(batch['text_observed'].any(1), g, torch.zeros_like(g))
                    av = batch['audio_observed'].any(1) | batch['vision_observed'].any(1)
                    g = torch.where(av, g, torch.ones_like(g))
                    assert torch.allclose(g.mean(), aux['gate_mean'], atol=1e-6)
                    gates.append(g.cpu().numpy());logits.append(out);raw.append(strength)
                rows.append(np.stack(gates, axis=1))
                predictions.append(torch.stack(logits).mean(0).argmax(-1).cpu().numpy())
                strengths.append(torch.stack(raw).mean(0).cpu().numpy())
        gates = np.concatenate(rows);mean = gates.mean(1)
        performance = metrics(np.asarray(fields['classification_labels']).ravel(), np.asarray(fields['regression_labels']).ravel(),np.concatenate(predictions), np.concatenate(strengths))
        result['scenarios'][mod or 'complete'] = dict(fraction=.4 if mod else 0, position='middle' if mod else None,
            gates=gates.tolist(), sample_mean=mean.tolist(), mean=float(mean.mean()), median=float(np.median(mean)),
            q25=float(np.quantile(mean,.25)), q75=float(np.quantile(mean,.75)), seed_means=gates.mean(0).tolist(), metrics=performance)
        print(mod or 'complete', performance['macro_f1'], float(mean.mean()), flush=True)
    reference=json.loads((folder/'gate_recon_ensemble.json').read_text(encoding="utf-8"))
    for mod,item in result['scenarios'].items():
        expected=reference['complete'] if mod=='complete' else next(x for x in reference['scenarios'] if x['modality']==mod and x['fraction']==.4 and x['position']=='middle')
        for key,value in item['metrics'].items():
            assert np.allclose(value,expected[key],atol=2e-6), (mod,key,value,expected[key])
    (folder/'gate_response_diagnostics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n', encoding="utf-8")

if __name__ == '__main__':
    main()
