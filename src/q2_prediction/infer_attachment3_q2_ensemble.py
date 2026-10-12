# 本程序及代码是在人工智能工具辅助下完成的
# 人工智能工具名称：ChatGPT，型号：GPT-6 Astra，开发公司：OpenAI，版本颁布日期：2026-09-04

"""Final aligned attachment-3 inference with frozen Q2 three-seed ensemble."""

from __future__ import annotations
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
for _p in (str(HERE.parent), str(HERE)):          # src/ 与本文件所在目录
    if _p not in sys.path:
        sys.path.insert(0, _p)


import csv
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel

from experiment_q2_mask_fusion import Specialist
from common.prep import BERT, consistent_strength, make_split

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data/attachment3/aligned"
OUT = ROOT / "outputs/q2_prediction"
SEEDS = (2026, 2027, 2028)
MODE = "gate_recon"
OUTPUT = OUT / "attachment3_predictions_gate_recon_ensemble.csv"
MANIFEST = OUT / "attachment3_gate_recon_ensemble_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = [OUT / f"q2_systematic/{MODE}_seed{seed}.pt" for seed in SEEDS]
    saves = [torch.load(path, map_location=device, weights_only=False) for path in paths]
    for saved in saves:
        if saved["mode"] != MODE or saved["data_version"] != "aligned_50":
            raise ValueError("Invalid frozen ensemble checkpoint")
    for saved in saves[1:]:
        for name in ("audio", "vision"):
            for a, b in zip(saves[0]["normalizers"][name], saved["normalizers"][name]):
                if not np.allclose(a, b):
                    raise ValueError("Normalizer mismatch")
    models = []
    for saved in saves:
        model = Specialist(MODE).to(device).eval()
        model.load_state_dict(saved["state_dict"])
        models.append(model)
    bert = AutoModel.from_pretrained(str(BERT), local_files_only=True).to(device).eval()
    bert.requires_grad_(False)
    paths_in = sorted(INPUT.glob("[0-9][0-9].pkl"))
    expected = [f"{n:02d}" for n in range(1, 31)]
    if [p.stem for p in paths_in] != expected:
        raise ValueError("Expected exactly attachment-3 aligned files 01.pkl–30.pkl")
    rows = []
    with torch.no_grad():
        for path in paths_in:
            with path.open("rb") as handle:
                fields = pickle.load(handle)["test"]
            if (np.asarray(fields["text_bert"]).shape != (1, 3, 50)
                    or np.asarray(fields["audio"]).shape != (1, 50, 74)
                    or np.asarray(fields["vision"]).shape != (1, 50, 35)):
                raise ValueError(f"Unexpected feature shape: {path.name}")
            batch = make_split(fields, saves[0]["normalizers"], device)
            predictions = [model(batch, bert) for model in models]
            logits = torch.stack([p[0] for p in predictions]).mean(0)
            strength_raw = torch.stack([p[1] for p in predictions]).mean(0)
            polarity = int(logits.argmax(-1).item())
            strength = float(consistent_strength(
                np.asarray([polarity]), strength_raw.cpu().numpy())[0])
            rows.append({"sample_id": path.stem, "polarity": polarity,
                         "strength": f"{strength:.6f}"})
    if len(rows) != 30 or len({r["sample_id"] for r in rows}) != 30:
        raise AssertionError("Expected 30 unique predictions")
    if not all(r["polarity"] in (0, 1, 2) and -3 <= float(r["strength"]) <= 3
               for r in rows):
        raise AssertionError("Output outside label range")
    if not all((r["polarity"] == 1 and float(r["strength"]) == 0)
               or (r["polarity"] == 0 and float(r["strength"]) < 0)
               or (r["polarity"] == 2 and float(r["strength"]) > 0)
               for r in rows):
        raise AssertionError("Class/strength sign mismatch")
    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "polarity", "strength"])
        writer.writeheader()
        writer.writerows(rows)
    manifest = {"data_version": "aligned_50", "mode": MODE, "seeds": SEEDS,
                "input_ids": expected, "n_predictions": len(rows),
                "checkpoints": {str(seed): {"path": str(path.relative_to(ROOT)),
                                            "sha256": sha256(path)}
                                for seed, path in zip(SEEDS, paths)},
                "ensemble_rule": "mean logits and raw strength; class/sign projection",
                "output": str(OUTPUT.relative_to(ROOT)),
                "output_sha256": sha256(OUTPUT),
                "label_mapping": {"0": "Negative", "1": "Neutral", "2": "Positive"}}
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": manifest["output"], "rows": len(rows),
                      "sha256": manifest["output_sha256"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
