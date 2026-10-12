# 本程序及代码是在人工智能工具辅助下完成的
# 人工智能工具名称：ChatGPT，型号：GPT-6 Astra，开发公司：OpenAI，版本颁布日期：2026-09-04

"""Shared audio/video encoder and temporal-fusion comparison model.

Train the fixed gate_recon model with q2_prediction/experiment_q2_mask_fusion.py.
Running this module trains the TemporalFusion comparison model.
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
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from torch import nn
from transformers import AutoModel

from prep import (BERT, DATA, augment, consistent_strength,
                                  make_split, moments, seed_all, slice_batch,
                                  train_normalizer, apply_missing)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/q2_prediction/comparisons"


class AVEncoder(nn.Module):
    def __init__(self, input_dim: int, width: int = 96) -> None:
        super().__init__()
        self.project = nn.Sequential(nn.Linear(input_dim, width), nn.LayerNorm(width), nn.GELU())
        self.position = nn.Parameter(torch.zeros(1, 50, width))
        self.conv = nn.Conv1d(width, width, kernel_size=3, padding=1)
        self.score = nn.Linear(width, 1)

    def forward(self, x: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
        h = self.project(x) + self.position[:, :x.shape[1]]
        h = h * observed.unsqueeze(-1)
        h = torch.nn.functional.gelu(h + self.conv(h.transpose(1, 2)).transpose(1, 2))
        h = h * observed.unsqueeze(-1)
        count = observed.sum(1, keepdim=True).clamp_min(1)
        mean = h.sum(1) / count
        score = self.score(h).squeeze(-1).masked_fill(~observed, -1e4)
        active = observed.any(1)
        score[~active, 0] = 0
        attention = torch.softmax(score, dim=1)
        pooled = (attention.unsqueeze(-1) * h).sum(1)
        pooled = pooled * active.unsqueeze(-1)
        fraction = observed.float().mean(1, keepdim=True)
        return torch.cat([mean, pooled, fraction], dim=-1)


class TemporalFusion(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.audio_encoder = AVEncoder(74)
        self.vision_encoder = AVEncoder(35)
        width = 1536 + (74 * 4 + 1) + (35 * 4 + 1) + 193 * 2
        self.shared = nn.Sequential(nn.Linear(width, 256), nn.LayerNorm(256), nn.GELU(),
                                    nn.Dropout(0.35), nn.Linear(256, 128), nn.GELU(),
                                    nn.Dropout(0.2))
        self.classifier = nn.Linear(128, 3)
        self.regressor = nn.Linear(128, 1)

    def forward(self, batch: dict, bert: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            hidden = bert(input_ids=batch["tokens"],
                          attention_mask=batch["struct"].long(),
                          token_type_ids=batch["segment"]).last_hidden_state.float()
        observed = batch["text_observed"].unsqueeze(-1).float()
        text_mean = (hidden * observed).sum(1) / observed.sum(1).clamp_min(1)
        text = torch.cat([hidden[:, 0], text_mean], dim=-1)
        audio = batch["audio"]
        vision = batch["vision"]
        am = batch["audio_observed"]
        vm = batch["vision_observed"]
        fused = torch.cat([text, moments(audio, am), moments(vision, vm),
                           self.audio_encoder(audio, am), self.vision_encoder(vision, vm)], dim=-1)
        h = self.shared(fused)
        return self.classifier(h), self.regressor(h).squeeze(-1)


def evaluate(split: dict, bert: nn.Module, model: TemporalFusion, batch_size: int,
             scenario: tuple[str, float, str] | None = None) -> dict:
    model.eval()
    classes, strengths = [], []
    with torch.no_grad():
        for first in range(0, len(split["class"]), batch_size):
            ids = np.arange(first, min(first + batch_size, len(split["class"])))
            batch = slice_batch(split, ids)
            if scenario:
                apply_missing(batch, *scenario)
            logits, reg = model(batch, bert)
            classes.append(logits.argmax(-1).cpu().numpy())
            strengths.append(reg.cpu().numpy())
    y = split["class"].cpu().numpy()
    r = split["strength"].cpu().numpy()
    pred = np.concatenate(classes)
    reg = np.concatenate(strengths)
    consistent = consistent_strength(pred, reg)
    return {"accuracy": float(accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, labels=[0, 1, 2], average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y, pred, labels=[0, 1, 2], average="weighted", zero_division=0)),
            "per_class_f1": f1_score(y, pred, labels=[0, 1, 2], average=None, zero_division=0).tolist(),
            "mae": float(mean_absolute_error(r, reg)),
            "pearson": float(np.corrcoef(r, reg)[0, 1]) if np.std(reg) else 0.0,
            "mae_consistent": float(mean_absolute_error(r, consistent)),
            "pearson_consistent": float(np.corrcoef(r, consistent)[0, 1]) if np.std(consistent) else 0.0}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--missing-prob", type=float, default=0.7)
    parser.add_argument("--full-text-drop-prob", type=float, default=0.0)
    parser.add_argument("--output-prefix", default="temporal_fusion")
    args = parser.parse_args()
    if not 0 <= args.missing_prob <= 1 or not 0 <= args.full_text_drop_prob <= 1:
        parser.error("probabilities must be in [0,1]")
    seed_all(args.seed)
    rng = random.Random(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with DATA.open("rb") as f:
        raw = pickle.load(f)
    normalizers = {name: train_normalizer(raw["train"], name) for name in ("audio", "vision")}
    train = make_split(raw["train"], normalizers, device)
    valid = make_split(raw["valid"], normalizers, device)
    bert = AutoModel.from_pretrained(str(BERT), local_files_only=True).to(device).eval()
    bert.requires_grad_(False)
    model = TemporalFusion().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    freq = torch.bincount(train["class"], minlength=3).float()
    weights = (1 / freq.sqrt())
    weights /= weights.mean()
    history = []
    best_score, patience = -1e9, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = np.random.permutation(len(train["class"]))
        losses = []
        for first in range(0, len(order), args.batch_size):
            batch = slice_batch(train, order[first:first + args.batch_size])
            augment(batch, rng, args.missing_prob)
            if args.full_text_drop_prob:
                selected = np.asarray([i for i in range(len(batch["tokens"]))
                                       if rng.random() < args.full_text_drop_prob], dtype=int)
                if len(selected):
                    apply_missing(batch, "text", 1.0, "middle", rows=selected)
            logits, regression = model(batch, bert)
            loss = nn.functional.cross_entropy(logits, batch["class"], weight=weights)
            loss += 0.2 * nn.functional.smooth_l1_loss(regression, batch["strength"])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        metric = evaluate(valid, bert, model, args.batch_size)
        score = metric["macro_f1"] + 0.1 * metric["pearson"] - 0.1 * metric["mae"]
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "joint_score": score,
               **metric}
        history.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if score > best_score:
            best_score, patience = score, 0
            torch.save({"state_dict": model.state_dict(), "normalizers": normalizers,
                        "data_version": "aligned_50", "seed": args.seed,
                        "epoch": epoch, "validation": metric,
                        "missing_probability": args.missing_prob,
                        "full_text_drop_probability": args.full_text_drop_prob},
                       OUT / f"{args.output_prefix}_best.pt")
        else:
            patience += 1
            if patience >= 6:
                break
    selected = torch.load(OUT / f"{args.output_prefix}_best.pt", map_location=device,
                          weights_only=False)
    model.load_state_dict(selected["state_dict"])
    scenarios = []
    for modality in ("text", "audio", "vision"):
        for fraction in (0.2, 0.4):
            for position in ("start", "middle", "end"):
                metric = evaluate(valid, bert, model, args.batch_size,
                                  (modality, fraction, position))
                scenarios.append({"modality": modality, "fraction": fraction,
                                  "position": position, **metric})
        metric = evaluate(valid, bert, model, args.batch_size, (modality, 1.0, "middle"))
        scenarios.append({"modality": modality, "fraction": 1.0,
                          "position": "middle", **metric})
    report = {"data_version": "aligned_50", "train_n": len(train["class"]),
              "valid_n": len(valid["class"]), "seed": args.seed,
              "missing_probability": args.missing_prob,
              "full_text_drop_probability": args.full_text_drop_prob,
              "selected_epoch": selected["epoch"],
              "selected_validation": selected["validation"],
              "history": history, "scenarios": scenarios}
    (OUT / f"{args.output_prefix}_results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("SELECTED", json.dumps({"epoch": selected["epoch"],
                                 **selected["validation"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
