# 本程序及代码是在人工智能工具辅助下完成的
# 人工智能工具名称：ChatGPT，型号：GPT-6 Astra，开发公司：OpenAI，版本颁布日期：2026-09-04

"""Attachment-2-only comparison of audio/visual experts and mask-aware fusion.

The gate_recon mode defines the fixed submission model; other modes are component comparisons.
Selection uses only train/valid of aligned_50; attachment 2 test and attachments
3/4 are never opened. All reported gaps are structural positions, not padding.
"""

from __future__ import annotations
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
for _p in (str(HERE.parent), str(HERE)):          # src/ 与本文件所在目录
    if _p not in sys.path:
        sys.path.insert(0, _p)


import sys
from pathlib import Path
SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

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

from common.prep import (BERT, DATA, apply_missing, augment,
                                  consistent_strength, make_split, missing_span,
                                  seed_all, slice_batch, train_normalizer)
from common.encoders import AVEncoder

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/q2_prediction/q2_systematic"


class MaskedBiLSTMEncoder(nn.Module):
    """BiLSTM counterpart to the second reference repo, on real 50-step features."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.project = nn.Sequential(nn.Linear(input_dim, 64), nn.LayerNorm(64), nn.GELU())
        self.lstm = nn.LSTM(64, 32, bidirectional=True, batch_first=True)
        self.score = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
        x = self.project(x) * observed.unsqueeze(-1)
        h, _ = self.lstm(x)
        h = h * observed.unsqueeze(-1)
        mean = h.sum(1) / observed.sum(1, keepdim=True).clamp_min(1)
        scores = self.score(h).squeeze(-1).masked_fill(~observed, -1e4)
        active = observed.any(1)
        scores = torch.where(active[:, None], scores, torch.zeros_like(scores))
        pooled = (torch.softmax(scores, dim=1).unsqueeze(-1) * h).sum(1)
        pooled = pooled * active.unsqueeze(-1)
        return torch.cat([mean, pooled, observed.float().mean(1, keepdim=True)], -1)


class Specialist(nn.Module):
    def __init__(self, mode: str):
        super().__init__()
        self.mode = mode
        encoder = MaskedBiLSTMEncoder if mode == "av_lstm" else lambda dim: AVEncoder(dim, 64)
        self.audio = encoder(74)
        self.vision = encoder(35)
        av_width = 129 * 2
        self.av_backbone = nn.Sequential(nn.Linear(av_width, 192), nn.LayerNorm(192),
                                         nn.GELU(), nn.Dropout(.25), nn.Linear(192, 128),
                                         nn.GELU())
        self.av_class = nn.Linear(128, 3)
        self.av_reg = nn.Linear(128, 1)
        if mode not in ("av", "av_lstm"):
            self.text_backbone = nn.Sequential(nn.Linear(1536, 192), nn.LayerNorm(192),
                                               nn.GELU(), nn.Dropout(.25),
                                               nn.Linear(192, 128), nn.GELU())
            self.text_class = nn.Linear(128, 3)
            self.text_reg = nn.Linear(128, 1)
            self.gate = nn.Sequential(nn.Linear(128 * 2 + 3, 64), nn.GELU(),
                                      nn.Linear(64, 1))
        if mode == "gate_recon":
            self.reconstruct = nn.Linear(128, 1536)
        if mode == "cross_attn":
            self.text_token = nn.Linear(1536, 128)
            self.audio_token = nn.Linear(129, 128)
            self.vision_token = nn.Linear(129, 128)
            layer = nn.TransformerEncoderLayer(128, 4, 256, .2, batch_first=True)
            self.cross_encoder = nn.TransformerEncoder(layer, 2)
            self.cross_class = nn.Linear(128, 3)
            self.cross_reg = nn.Linear(128, 1)

    def forward(self, batch: dict, bert: nn.Module | None):
        audio = self.audio(batch["audio"], batch["audio_observed"])
        vision = self.vision(batch["vision"], batch["vision_observed"])
        av_h = self.av_backbone(torch.cat([audio, vision], dim=-1))
        av_logits, av_reg = self.av_class(av_h), self.av_reg(av_h).squeeze(-1)
        if self.mode in ("av", "av_lstm"):
            return av_logits, av_reg, {"av_logits": av_logits, "av_reg": av_reg}
        with torch.no_grad():
            h = bert(input_ids=batch["tokens"], attention_mask=batch["struct"].long(),
                     token_type_ids=batch["segment"]).last_hidden_state.float()
            w = batch["text_observed"].unsqueeze(-1).float()
            text = torch.cat([h[:, 0], (h * w).sum(1) / w.sum(1).clamp_min(1)], dim=-1)
        if self.mode == "cross_attn":
            tokens = torch.stack([self.text_token(text), self.audio_token(audio),
                                  self.vision_token(vision)], dim=1)
            active = torch.stack([batch[f"{m}_observed"].any(1)
                                  for m in ("text", "audio", "vision")], dim=1)
            mixed = self.cross_encoder(tokens, src_key_padding_mask=~active)
            pooled = (mixed * active.unsqueeze(-1)).sum(1) / active.sum(1, keepdim=True).clamp_min(1)
            logits, reg = self.cross_class(pooled), self.cross_reg(pooled).squeeze(-1)
            return logits, reg, {"av_logits": av_logits, "av_reg": av_reg}
        text_h = self.text_backbone(text)
        text_logits, text_reg = self.text_class(text_h), self.text_reg(text_h).squeeze(-1)
        fractions = torch.stack([batch[f"{m}_observed"].float().mean(1)
                                 for m in ("text", "audio", "vision")], dim=-1)
        g = torch.sigmoid(self.gate(torch.cat([text_h, av_h, fractions], dim=-1))).squeeze(-1)
        av_available = (fractions[:, 1] + fractions[:, 2]) > 0
        text_available = fractions[:, 0] > 0
        g = torch.where(text_available, g, torch.zeros_like(g))
        g = torch.where(av_available, g, torch.ones_like(g))
        logits = g[:, None] * text_logits + (1 - g[:, None]) * av_logits
        reg = g * text_reg + (1 - g) * av_reg
        aux = {"av_logits": av_logits, "av_reg": av_reg,
               "text_logits": text_logits, "text_reg": text_reg,
               "text_available": text_available, "gate_mean": g.mean()}
        if self.mode == "gate_recon":
            aux["reconstructed_text"] = self.reconstruct(av_h)
        return logits, reg, aux


def av_augment(batch: dict, rng: random.Random):
    for row in range(len(batch["tokens"])):
        if rng.random() >= .7:
            continue
        chosen = rng.sample(("audio", "vision"), 2 if rng.random() < .2 else 1)
        for modality in chosen:
            eligible = batch["struct"][row] & (batch["tokens"][row] != 0)
            eligible &= (batch["tokens"][row] != 101) & (batch["tokens"][row] != 102)
            n = int(eligible.sum())
            if n == 0:
                continue
            frac = rng.uniform(.15, .4)
            gap = missing_span(eligible[None], frac, "random", rng)[0]
            gap &= batch[f"{modality}_observed"][row]
            batch[f"{modality}_observed"][row, gap] = False
            batch[modality][row, gap] = 0


def metrics(y, r, p, z):
    c = consistent_strength(p, z)
    return {"accuracy": float(accuracy_score(y, p)),
            "macro_f1": float(f1_score(y, p, labels=[0, 1, 2], average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y, p, labels=[0, 1, 2], average="weighted", zero_division=0)),
            "per_class_f1": f1_score(y, p, labels=[0, 1, 2], average=None, zero_division=0).tolist(),
            "mae": float(mean_absolute_error(r, z)),
            "pearson": float(np.corrcoef(r, z)[0, 1]) if np.std(z) else 0.,
            "mae_consistent": float(mean_absolute_error(r, c)),
            "pearson_consistent": float(np.corrcoef(r, c)[0, 1]) if np.std(c) else 0.}


def evaluate(split, model, bert, batch_size, scenario=None):
    model.eval()
    pred, strength, gates = [], [], []
    with torch.no_grad():
        for start in range(0, len(split["class"]), batch_size):
            ids = np.arange(start, min(start + batch_size, len(split["class"])))
            batch = slice_batch(split, ids)
            if scenario:
                for spec in scenario:
                    apply_missing(batch, *spec)
            logits, reg, aux = model(batch, bert)
            pred.append(logits.argmax(-1).cpu().numpy())
            strength.append(reg.cpu().numpy())
            if "gate_mean" in aux:
                gates.append(float(aux["gate_mean"]))
    result = metrics(split["class"].cpu().numpy(), split["strength"].cpu().numpy(),
                     np.concatenate(pred), np.concatenate(strength))
    if gates:
        result["mean_text_gate"] = float(np.mean(gates))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("av", "av_lstm", "gate", "gate_recon", "cross_attn"), required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--neutral-loss-weight", type=float, default=0.)
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    OUT.mkdir(parents=True, exist_ok=True)
    seed_all(args.seed)
    rng = random.Random(args.seed)
    device = torch.device(args.device)
    with DATA.open("rb") as handle:
        raw = pickle.load(handle)
    normalizers = {name: train_normalizer(raw["train"], name) for name in ("audio", "vision")}
    train = make_split(raw["train"], normalizers, device)
    valid = make_split(raw["valid"], normalizers, device)
    del raw
    bert = None if args.mode in ("av", "av_lstm") else AutoModel.from_pretrained(
        str(BERT), local_files_only=True).to(device).eval().requires_grad_(False)
    model = Specialist(args.mode).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
    counts = torch.bincount(train["class"], minlength=3).float()
    weights = 1 / counts.sqrt()
    weights /= weights.mean()
    history, best_score, patience = [], -1e9, 0
    if args.neutral_loss_weight < 0:
        parser.error("--neutral-loss-weight must be nonnegative")
    stem = f"{args.mode}{'_neutral' if args.neutral_loss_weight else ''}_seed{args.seed}"
    checkpoint = OUT / f"{stem}.pt"
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = np.random.permutation(len(train["class"]))
        losses = []
        for start in range(0, len(order), args.batch_size):
            batch = slice_batch(train, order[start:start + args.batch_size])
            if args.mode in ("av", "av_lstm"):
                av_augment(batch, rng)
            else:
                augment(batch, rng)
                selected = np.asarray([i for i in range(len(batch["tokens"]))
                                       if rng.random() < .1], dtype=int)
                if len(selected):
                    apply_missing(batch, "text", 1., "middle", rows=selected)
            logits, reg, aux = model(batch, bert)
            loss = nn.functional.cross_entropy(logits, batch["class"], weight=weights)
            loss += .2 * nn.functional.smooth_l1_loss(reg, batch["strength"])
            if args.neutral_loss_weight:
                neutral_logit = logits[:, 1] - torch.logsumexp(logits[:, [0, 2]], dim=-1)
                neutral_target = (batch["class"] == 1).float()
                neutral_count = counts[1]
                neutral_pos_weight = (counts.sum() - neutral_count) / neutral_count
                loss += args.neutral_loss_weight * nn.functional.binary_cross_entropy_with_logits(
                    neutral_logit, neutral_target, pos_weight=neutral_pos_weight)
            if args.mode in ("gate", "gate_recon"):
                av_loss = nn.functional.cross_entropy(aux["av_logits"], batch["class"], weight=weights)
                av_loss += .2 * nn.functional.smooth_l1_loss(aux["av_reg"], batch["strength"])
                loss += .3 * av_loss
                available = aux["text_available"]
                if available.any():
                    text_loss = nn.functional.cross_entropy(aux["text_logits"][available], batch["class"][available], weight=weights)
                    text_loss += .2 * nn.functional.smooth_l1_loss(aux["text_reg"][available], batch["strength"][available])
                    loss += .2 * text_loss
                if args.mode == "gate_recon":
                    # Reconstruction is trained only on clean text samples. The target
                    # is the frozen BERT representation, never a validation embedding.
                    clean = batch["text_observed"].sum(1) == (
                        batch["struct"].sum(1) - 2)
                    if clean.any():
                        with torch.no_grad():
                            h = bert(input_ids=batch["tokens"][clean],
                                     attention_mask=batch["struct"][clean].long(),
                                     token_type_ids=batch["segment"][clean]).last_hidden_state.float()
                            w = batch["text_observed"][clean].unsqueeze(-1).float()
                            target = torch.cat([h[:, 0], (h * w).sum(1) / w.sum(1).clamp_min(1)], -1)
                        loss += .02 * nn.functional.mse_loss(aux["reconstructed_text"][clean], target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        full = evaluate(valid, model, bert, args.batch_size)
        text_mid = evaluate(valid, model, bert, args.batch_size, [("text", .4, "middle")])
        score = full["macro_f1"] + .25 * text_mid["macro_f1"] + .1 * full["pearson"] - .1 * full["mae"]
        record = {"epoch": epoch, "train_loss": float(np.mean(losses)),
                  "selection_score": float(score), "complete": full,
                  "text_middle_40": text_mid}
        history.append(record)
        print(json.dumps({"mode": args.mode, **record}, ensure_ascii=False), flush=True)
        if score > best_score:
            best_score, patience = score, 0
            torch.save({"state_dict": model.state_dict(), "normalizers": normalizers,
                        "mode": args.mode, "seed": args.seed, "epoch": epoch,
                        "data_version": "aligned_50", "selection_score": score}, checkpoint)
        else:
            patience += 1
            if patience >= 4:
                break
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(saved["state_dict"])
    scenarios = []
    for modality in ("text", "audio", "vision"):
        for fraction in (.2, .4):
            for position in ("start", "middle", "end"):
                scenarios.append({"modality": modality, "fraction": fraction,
                                  "position": position,
                                  **evaluate(valid, model, bert, args.batch_size,
                                             [(modality, fraction, position)])})
        scenarios.append({"modality": modality, "fraction": 1., "position": "middle",
                          **evaluate(valid, model, bert, args.batch_size,
                                     [(modality, 1., "middle")])})
    report = {"mode": args.mode, "seed": args.seed,
              "neutral_loss_weight": args.neutral_loss_weight,
              "train_n": len(train["class"]),
              "valid_n": len(valid["class"]), "data_version": "aligned_50",
              "selection_rule": "full_macro_f1 + .25*text_mid40_macro_f1 + .1*full_pearson - .1*full_mae",
              "selected_epoch": saved["epoch"], "history": history,
              "complete": evaluate(valid, model, bert, args.batch_size), "scenarios": scenarios,
              "checkpoint": str(checkpoint.relative_to(ROOT))}
    (OUT / f"{stem}.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("SELECTED", json.dumps({"mode": args.mode, "epoch": saved["epoch"],
                                  "complete": report["complete"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
