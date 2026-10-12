# 本程序及代码是在人工智能工具辅助下完成的
# 人工智能工具名称：ChatGPT，型号：GPT-6 Astra，开发公司：OpenAI，版本颁布日期：2026-09-04

"""Evaluate a fixed three-seed Q2 ensemble on attachment 2 validation or test split.

默认在验证集上评估并输出 `outputs/q2_prediction/q2_systematic/<mode>_ensemble.json`；
用 `--split test` 在常规测试集上评估（仅用于方案冻结后的报告，不参与选模），
输出 `<mode>_ensemble_test.json`，并通过 `--scenarios` 可选地在测试集上重复缺失场景。
"""

from __future__ import annotations
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
for _p in (str(HERE.parent), str(HERE)):          # src/ 与本文件所在目录
    if _p not in sys.path:
        sys.path.insert(0, _p)


import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel

from experiment_q2_mask_fusion import Specialist, metrics
from common.prep import BERT, DATA, apply_missing, make_split, slice_batch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/q2_prediction/q2_systematic"
SEEDS = (2026, 2027, 2028)


def evaluate(split, models, bert, batch_size, scenario=None):
    classes, strengths = [], []
    with torch.no_grad():
        for start in range(0, len(split["class"]), batch_size):
            ids = np.arange(start, min(start + batch_size, len(split["class"])))
            batch = slice_batch(split, ids)
            if scenario:
                for spec in scenario:
                    apply_missing(batch, *spec)
            predictions = [model(batch, bert) for model in models]
            logits = torch.stack([p[0] for p in predictions]).mean(0)
            reg = torch.stack([p[1] for p in predictions]).mean(0)
            classes.append(logits.argmax(-1).cpu().numpy())
            strengths.append(reg.cpu().numpy())
    return metrics(split["class"].cpu().numpy(), split["strength"].cpu().numpy(),
                   np.concatenate(classes), np.concatenate(strengths))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("gate", "gate_recon"), required=True)
    parser.add_argument("--split", choices=("valid", "test"), default="valid",
                        help="valid=验证集（默认，选模用）；test=常规测试集（冻结后报告用）")
    parser.add_argument("--scenarios", action="store_true",
                        help="是否同时评估 21 个缺失场景（默认仅在验证集上评估）")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    saves = [torch.load(OUT / f"{args.mode}_seed{seed}.pt", map_location=device,
                        weights_only=False) for seed in SEEDS]
    for s in saves:
        if s["data_version"] != "aligned_50" or s["mode"] != args.mode:
            raise ValueError("Incompatible checkpoint")
    for other in saves[1:]:
        for name in ("audio", "vision"):
            for a, b in zip(saves[0]["normalizers"][name], other["normalizers"][name]):
                if not np.allclose(a, b):
                    raise ValueError("Normalizer mismatch")
    with DATA.open("rb") as handle:
        fields = pickle.load(handle)[args.split]
    evaluated = make_split(fields, saves[0]["normalizers"], device)
    run_scenarios = args.scenarios or args.split == "valid"
    bert = AutoModel.from_pretrained(str(BERT), local_files_only=True).to(device).eval()
    bert.requires_grad_(False)
    models = []
    for saved in saves:
        model = Specialist(args.mode).to(device).eval()
        model.load_state_dict(saved["state_dict"])
        models.append(model)
    complete = evaluate(evaluated, models, bert, 64)
    scenarios = []
    for modality in (("text", "audio", "vision") if run_scenarios else ()):
        for fraction in (.2, .4):
            for position in ("start", "middle", "end"):
                scenarios.append({"modality": modality, "fraction": fraction,
                                  "position": position,
                                  **evaluate(evaluated, models, bert, 64,
                                             [(modality, fraction, position)])})
        scenarios.append({"modality": modality, "fraction": 1., "position": "middle",
                          **evaluate(evaluated, models, bert, 64,
                                     [(modality, 1., "middle")])})
    report = {"mode": args.mode, "seeds": SEEDS, "data_version": "aligned_50",
              "split": args.split, "train_n": 3395, f"{args.split}_n": len(evaluated["class"]),
              "ensemble_rule": "arithmetic mean of class logits and raw strength from three train seeds",
              "complete": complete, "scenarios": scenarios}
    suffix = "" if args.split == "valid" else "_test"
    output = OUT / f"{args.mode}_ensemble{suffix}.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"split": args.split, "output": str(output.relative_to(ROOT)), "complete": complete},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
