# 本程序及代码是在人工智能工具辅助下完成的
# 人工智能工具名称：ChatGPT，型号：GPT-6 Astra，开发公司：OpenAI，版本颁布日期：2026-09-04

"""Q3 explanations for the frozen aligned Q2 ensemble.

Validation uses attachment-2 valid only. Attachment-4 is read only after the
explanation rule is fixed. Occlusion measures model sensitivity, not causality.
"""

from __future__ import annotations
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
for _p in (str(HERE.parent), str(HERE)):          # src/ 与本文件所在目录
    if _p not in sys.path:
        sys.path.insert(0, _p)


import argparse
import csv
import hashlib
import json
import pickle

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from q2_prediction.experiment_q2_mask_fusion import Specialist
from common.prep import BERT, DATA, consistent_strength, make_split, slice_batch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/q3_explanation/q3"
NAMES = ("text", "audio", "vision")
SEEDS = (2026, 2027, 2028)
WINDOW_FRACTION = .2
VALIDATION_COUNT = 60
RANDOM_SEED = 2026


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model(device):
    paths = [ROOT / f"outputs/q2_prediction/q2_systematic/gate_recon_seed{s}.pt" for s in SEEDS]
    saves = [torch.load(p, map_location=device, weights_only=False) for p in paths]
    for saved in saves:
        if saved["mode"] != "gate_recon" or saved["data_version"] != "aligned_50":
            raise ValueError("Wrong frozen Q2 checkpoint")
    models = []
    for saved in saves:
        model = Specialist("gate_recon").to(device).eval()
        model.load_state_dict(saved["state_dict"])
        models.append(model)
    bert = AutoModel.from_pretrained(str(BERT), local_files_only=True).to(device).eval()
    bert.requires_grad_(False)
    return saves[0]["normalizers"], models, bert, paths


def one(batch, index):
    return {k: v[index:index + 1].clone() for k, v in batch.items()}


def hide(batch, name, positions):
    key = f"{name}_observed"
    positions = torch.as_tensor(positions, dtype=torch.long, device=batch["tokens"].device)
    observed = batch[key][0, positions]
    positions = positions[observed]
    batch[key][0, positions] = False
    if name == "text":
        batch["tokens"][0, positions] = 103
    else:
        batch[name][0, positions] = 0


def predict(variants, models, bert, chunk=32):
    outputs = []
    with torch.inference_mode():
        for first in range(0, len(variants), chunk):
            group = variants[first:first + chunk]
            batch = {key: torch.cat([row[key] for row in group], dim=0) for key in group[0]}
            preds = [model(batch, bert) for model in models]
            logits = torch.stack([p[0] for p in preds]).mean(0)
            strength = torch.stack([p[1] for p in preds]).mean(0)
            probabilities = torch.softmax(logits, dim=-1)
            for p, z in zip(probabilities.cpu().numpy(), strength.cpu().numpy()):
                outputs.append((p.astype(float), float(z)))
    return outputs


def observed_positions(batch, name):
    return torch.nonzero(batch[f"{name}_observed"][0], as_tuple=True)[0].cpu().numpy()


def coalition_explanation(batch, models, bert):
    variants = []
    masks = list(range(8))
    for mask in masks:
        variant = one(batch, 0)
        for bit, name in enumerate(NAMES):
            if not mask & (1 << bit):
                hide(variant, name, observed_positions(variant, name))
        variants.append(variant)
    predictions = predict(variants, models, bert)
    complete = predictions[7]
    target = int(np.argmax(complete[0]))
    values = np.array([p[target] for p, _ in predictions])
    strength_values = np.array([z for _, z in predictions])
    # 3-player Shapley weights: |S|!(2-|S|)! / 3!.
    weights = (1 / 3, 1 / 6, 1 / 3)
    effects, strength_effects = [], []
    for bit in range(3):
        terms, reg_terms = [], []
        for mask in masks:
            if mask & (1 << bit):
                continue
            weight = weights[mask.bit_count()]
            terms.append(weight * (values[mask | (1 << bit)] - values[mask]))
            reg_terms.append(weight * (strength_values[mask | (1 << bit)] - strength_values[mask]))
        effects.append(float(sum(terms)))
        strength_effects.append(float(sum(reg_terms)))
    if not np.isclose(sum(effects), values[7] - values[0], atol=1e-6):
        raise AssertionError("Shapley additivity failed")
    positive = np.maximum(effects, 0)
    relative = positive / positive.sum() if positive.sum() > 0 else np.zeros(3)
    strength = float(consistent_strength(np.array([target]), np.array([complete[1]]))[0])
    return {"polarity": target, "strength": strength, "full_probability": float(values[7]),
            "empty_probability": float(values[0]), "shapley_probability": effects,
            "shapley_strength_raw": strength_effects, "positive_share": relative.tolist(),
            "main_modality": NAMES[int(np.argmax(effects))],
            "full_probabilities": complete[0].tolist()}


def local_explanation(batch, models, bert, target, name):
    positions = observed_positions(batch, name)
    if not len(positions):
        return None
    width = max(1, round(len(positions) * WINDOW_FRACTION))
    windows = [positions[start:start + width] for start in range(len(positions) - width + 1)]
    variants = []
    for window in windows:
        variant = one(batch, 0)
        hide(variant, name, window)
        variants.append(variant)
    base = predict([one(batch, 0)], models, bert)[0][0][target]
    masked = predict(variants, models, bert)
    drops = np.array([base - p[target] for p, _ in masked])
    high = int(np.argmax(drops))
    low = int(np.argmin(drops))
    # Derive a reproducible random-window seed from the token sum and modality name length.
    rng = np.random.default_rng(RANDOM_SEED + int(batch["tokens"][0].sum().item()) + len(name))
    random_idx = int(rng.integers(len(windows)))
    def record(index):
        return {"start_position": int(windows[index][0]),
                "end_position": int(windows[index][-1]),
                "n_observed_positions": int(len(windows[index])),
                "confidence_drop": float(drops[index])}
    return {"high": record(high), "low": record(low), "random": record(random_idx),
            "n_candidate_windows": len(windows), "all_window_drops": drops.tolist()}


def explain(batch, models, bert):
    global_result = coalition_explanation(batch, models, bert)
    target = global_result["polarity"]
    local = {name: local_explanation(batch, models, bert, target, name) for name in NAMES}
    return {**global_result, "local": local}


def expanded_deletion(batch, models, bert, target, name, center_position):
    """Evaluate a wider 40% window, separate from the 20% scoring windows."""
    positions = observed_positions(batch, name)
    if not len(positions):
        return None
    width = max(1, round(len(positions) * .4))
    center = int(np.argmin(np.abs(positions - center_position)))
    start = min(max(0, center - width // 2), len(positions) - width)
    variant = one(batch, 0)
    hide(variant, name, positions[start:start + width])
    return predict([variant], models, bert)[0][0][target]


def validation(normalizers, models, bert, full=False):
    with DATA.open("rb") as handle:
        fields = pickle.load(handle)["valid"]
    split = make_split(fields, normalizers, next(models[0].parameters()).device)
    indices = (np.arange(len(split["class"])) if full else
               np.linspace(0, len(split["class"]) - 1, VALIDATION_COUNT, dtype=int))
    details = []
    for count, idx in enumerate(indices, 1):
        batch = slice_batch(split, np.array([idx]))
        result = explain(batch, models, bert)
        # Scoring uses 20% windows. This check deletes 40% around the selected
        # center, so its observed drop is not the optimized score itself.
        transfer = {}
        for name in NAMES:
            loc = result["local"][name]
            if loc is None:
                continue
            base = result["full_probability"]
            transfer[name] = {}
            for level in ("high", "random", "low"):
                item = loc[level]
                center = (item["start_position"] + item["end_position"]) // 2
                prob = expanded_deletion(batch, models, bert, result["polarity"], name, center)
                transfer[name][level] = float(base - prob)
        result["wider_window_transfer"] = transfer
        truth = int(split["class"][idx].item())
        details.append({"valid_index": int(idx), "sample_id": str(fields["id"][idx]),
                        "true_polarity": truth, "prediction_correct": result["polarity"] == truth,
                        **result})
        if count % (50 if full else 10) == 0:
            print(f"validation {count}/{len(indices)}", flush=True)
    fidelity = {}
    for name in NAMES:
        rows = [r["local"][name] for r in details if r["local"][name] is not None]
        fidelity[name] = {"n": len(rows),
                          "mean_high_drop": float(np.mean([r["high"]["confidence_drop"] for r in rows])),
                          "mean_random_drop": float(np.mean([r["random"]["confidence_drop"] for r in rows])),
                          "mean_low_drop": float(np.mean([r["low"]["confidence_drop"] for r in rows])),
                          "fraction_high_gt_random": float(np.mean([r["high"]["confidence_drop"] > r["random"]["confidence_drop"] for r in rows]))}
        wider = [r["wider_window_transfer"][name] for r in details if name in r["wider_window_transfer"]]
        fidelity[name]["wider_40pct_mean_high_drop"] = float(np.mean([r["high"] for r in wider]))
        fidelity[name]["wider_40pct_mean_random_drop"] = float(np.mean([r["random"] for r in wider]))
        fidelity[name]["wider_40pct_mean_low_drop"] = float(np.mean([r["low"] for r in wider]))
        fidelity[name]["wider_40pct_fraction_high_gt_random"] = float(np.mean([r["high"] > r["random"] for r in wider]))
    strata = {}
    for label, subset in ([('correct', [r for r in details if r['prediction_correct']]),
                           ('incorrect', [r for r in details if not r['prediction_correct']])] +
                          [(f'true_class_{c}', [r for r in details if r['true_polarity'] == c])
                           for c in range(3)]):
        strata[label] = {'n': len(subset)}
        for name in NAMES:
            rows = [r['wider_window_transfer'][name] for r in subset
                    if name in r['wider_window_transfer']]
            strata[label][name] = {'n': len(rows),
                'mean_high_drop': float(np.mean([r['high'] for r in rows])) if rows else None,
                'mean_random_drop': float(np.mean([r['random'] for r in rows])) if rows else None,
                'fraction_high_gt_random': float(np.mean([r['high'] > r['random'] for r in rows])) if rows else None}
    report = {"data_version": "aligned_50", "split": "attachment2_valid",
              "n_total_valid": len(split["class"]), "n_explained": len(details),
              "indices": indices.tolist(), "window_fraction": WINDOW_FRACTION,
              "method": "8-coalition predicted-class probability Shapley; observed-position contiguous occlusion",
              "fidelity": fidelity, "strata": strata, "details": details}
    path = OUT / ("validation_explanation_all.json" if full else "validation_explanation.json")
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(path.relative_to(ROOT)), "fidelity": fidelity}, ensure_ascii=False))


def attachment4(normalizers, models, bert, checkpoints):
    source = ROOT / "data/attachment4/aligned"
    paths = sorted(source.glob("[0-9][0-9].pkl"))
    if [p.stem for p in paths] != [f"{i:02d}" for i in range(1, 21)]:
        raise ValueError("Expected attachment-4 files 01.pkl through 20.pkl")
    tokenizer = AutoTokenizer.from_pretrained(str(BERT), local_files_only=True)
    rows, details = [], []
    for count, path in enumerate(paths, 1):
        with path.open("rb") as handle:
            raw = pickle.load(handle)
        if str(raw["id"]) != path.stem:
            raise ValueError(f"ID mismatch: {path.name}")
        fields = {key: np.asarray(value)[None] for key, value in raw.items()
                  if key in ("text_bert", "audio", "vision")}
        batch = make_split(fields, normalizers, next(models[0].parameters()).device)
        result = explain(batch, models, bert)
        tokens = np.asarray(raw["text_bert"])[0]
        for name in NAMES:
            segment = result["local"][name]
            if segment and name == "text":
                start = segment["high"]["start_position"]
                end = segment["high"]["end_position"]
                segment["high"]["bert_wordpieces"] = tokenizer.convert_ids_to_tokens(tokens[start:end + 1].tolist())
        row = {"sample_id": path.stem, "polarity": result["polarity"],
               "strength": f"{result['strength']:.6f}",
               "main_modality": result["main_modality"],
               "text_contribution": f"{result['shapley_probability'][0]:.6f}",
               "audio_contribution": f"{result['shapley_probability'][1]:.6f}",
               "vision_contribution": f"{result['shapley_probability'][2]:.6f}",
               "text_share": f"{result['positive_share'][0]:.6f}",
               "audio_share": f"{result['positive_share'][1]:.6f}",
               "vision_share": f"{result['positive_share'][2]:.6f}",
               "key_modality": result["main_modality"],
               "key_start_position": result["local"][result["main_modality"]]["high"]["start_position"] if result["local"][result["main_modality"]] else "",
               "key_end_position": result["local"][result["main_modality"]]["high"]["end_position"] if result["local"][result["main_modality"]] else "",
               "key_confidence_drop": f"{result['local'][result['main_modality']]['high']['confidence_drop']:.6f}" if result["local"][result["main_modality"]] else "",
               "time_start_seconds": "", "time_end_seconds": "",
               "time_status": "feature_position_only_unverified_video_time"}
        rows.append(row)
        details.append({"sample_id": path.stem, "raw_text": str(raw["raw_text"]), **result})
        print(f"attachment4 {count}/20", flush=True)
    csv_path = OUT / "attachment4_predictions_explanations.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    details_path = OUT / "attachment4_explanations_full.json"
    details_path.write_text(json.dumps(details, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {"data_version": "aligned_50", "mode": "gate_recon", "seeds": SEEDS,
                "n_predictions": len(rows), "input_files": [p.name for p in paths],
                "checkpoint_sha256": {str(s): sha256(p) for s, p in zip(SEEDS, checkpoints)},
                "csv": str(csv_path.relative_to(ROOT)), "csv_sha256": sha256(csv_path),
                "details": str(details_path.relative_to(ROOT)),
                "time_status": "feature positions only; video seconds pending independent verification"}
    manifest_path = OUT / "attachment4_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(csv_path.relative_to(ROOT)), "rows": len(rows),
                      "sha256": manifest["csv_sha256"]}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("validation", "attachment4"), required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--full-validation", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    normalizers, models, bert, paths = load_model(torch.device(args.device))
    if args.stage == "validation":
        validation(normalizers, models, bert, full=args.full_validation)
    else:
        if not any((OUT / name).exists() for name in
                   ("validation_explanation_all.json", "validation_explanation.json")):
            raise RuntimeError("Run validation explanation before attachment-4 inference")
        attachment4(normalizers, models, bert, paths)


if __name__ == "__main__":
    main()
