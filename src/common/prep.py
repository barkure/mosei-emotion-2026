# 本程序及代码是在人工智能工具辅助下完成的
# 人工智能工具名称：ChatGPT，型号：GPT-6 Astra，开发公司：OpenAI，版本颁布日期：2026-09-04

"""Train an aligned three-modality probe with local span-missing augmentation.

Uses only attachment 2 train/valid. Frozen BERT is run on the masked token IDs,
so simulated text gaps do not retain the original token in its contextual states.
"""

from __future__ import annotations

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

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/attachment2/aligned_50.pkl"
BERT = ROOT / "models/bert-base-uncased"
OUT = ROOT / "outputs/q2_prediction/comparisons"
MODALITIES = ("text", "audio", "vision")


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_normalizer(fields: dict, name: str) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(fields[name], dtype=np.float32)
    token = np.asarray(fields["text_bert"])
    struct = token[:, 1, :].astype(bool)
    observed = struct & np.any(x != 0, axis=-1)
    rows = x[observed].astype(np.float64)
    mean = rows.mean(axis=0).astype(np.float32)
    std = np.maximum(rows.std(axis=0).astype(np.float32), 1e-4)
    return mean, std



def consistent_strength(classes: np.ndarray, strength: np.ndarray) -> np.ndarray:
    """Project strength onto the official class/sign convention.

    负向取 min(s, -1e-6)、中性取 0、正向取 max(s, 1e-6)，再 clip 到 [-3, 3]。
    """
    return np.where(classes == 1, 0.0,
                    np.where(classes == 0, np.minimum(strength, -1e-6),
                             np.maximum(strength, 1e-6))).clip(-3, 3)

def make_split(fields: dict, normalizers: dict, device: torch.device) -> dict:
    bert = np.asarray(fields["text_bert"])
    tokens = bert[:, 0, :].astype(np.int64)
    struct = bert[:, 1, :].astype(bool)
    text_observed = struct & (tokens != 0) & (tokens != 101) & (tokens != 102)
    batch = {
        "tokens": torch.as_tensor(tokens, dtype=torch.long, device=device),
        "segment": torch.as_tensor(bert[:, 2, :], dtype=torch.long, device=device),
        "struct": torch.as_tensor(struct, dtype=torch.bool, device=device),
        "text_observed": torch.as_tensor(text_observed, dtype=torch.bool, device=device),
        "class": torch.as_tensor(np.asarray(fields.get("classification_labels", np.zeros(len(tokens)))).reshape(-1),
                                 dtype=torch.long, device=device),
        "strength": torch.as_tensor(np.asarray(fields.get("regression_labels", np.zeros(len(tokens)))).reshape(-1),
                                    dtype=torch.float32, device=device),
    }
    for name in ("audio", "vision"):
        raw = np.asarray(fields[name], dtype=np.float32)
        observed = struct & np.any(raw != 0, axis=-1)
        mean, std = normalizers[name]
        values = np.clip((raw - mean) / std, -10, 10)
        values[~observed] = 0
        batch[name] = torch.as_tensor(values, dtype=torch.float32, device=device)
        batch[f"{name}_observed"] = torch.as_tensor(observed, dtype=torch.bool, device=device)
    return batch


def slice_batch(split: dict, ids: np.ndarray) -> dict:
    return {key: value[ids] for key, value in split.items()}


def missing_span(observed: torch.Tensor, fraction: float, position: str,
                 rng: random.Random | None = None) -> torch.Tensor:
    """Choose a contiguous interval in the valid observed prefix of each sample."""
    missing = torch.zeros_like(observed)
    for row in range(len(observed)):
        positions = torch.nonzero(observed[row], as_tuple=True)[0]
        n = len(positions)
        if n == 0:
            continue
        width = max(1, min(n, round(n * fraction)))
        if position == "start":
            start = 0
        elif position == "middle":
            start = (n - width) // 2
        elif position == "end":
            start = n - width
        elif position == "random" and rng is not None:
            start = rng.randint(0, n - width)
        else:
            raise ValueError(position)
        missing[row, positions[start:start + width]] = True
    return missing


def apply_missing(batch: dict, modality: str, fraction: float, position: str,
                  rng: random.Random | None = None, rows: np.ndarray | None = None) -> None:
    key = f"{modality}_observed"
    missing = missing_span(batch[key], fraction, position, rng)
    if rows is not None:
        selected = torch.zeros(len(missing), device=missing.device, dtype=torch.bool)
        selected[rows] = True
        missing &= selected[:, None]
    batch[key] &= ~missing
    if modality == "text":
        batch["tokens"][missing] = 103  # [MASK], keeping the structural attention mask.
    else:
        batch[modality][missing] = 0


def augment(batch: dict, rng: random.Random, missing_prob: float = 0.7) -> None:
    for row in range(len(batch["tokens"])):
        if rng.random() >= missing_prob:
            continue
        chosen = rng.sample(MODALITIES, k=2 if rng.random() < 0.2 else 1)
        for modality in chosen:
            apply_missing(batch, modality, rng.uniform(0.15, 0.4), "random", rng,
                          np.asarray([row]))


def moments(x: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
    w = observed.unsqueeze(-1).to(x.dtype)
    count = w.sum(1).clamp_min(1)
    mean = (x * w).sum(1) / count
    std = (((x - mean[:, None, :]) ** 2 * w).sum(1) / count).sqrt()
    half = x.shape[1] // 2
    halves = []
    for part, part_w in ((x[:, :half], w[:, :half]), (x[:, half:], w[:, half:])):
        halves.append((part * part_w).sum(1) / part_w.sum(1).clamp_min(1))
    fraction = observed.float().mean(1, keepdim=True)
    return torch.cat([mean, std, *halves, fraction], dim=-1)


class FusionHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        width = 1536 + (74 * 4 + 1) + (35 * 4 + 1)
        self.shared = nn.Sequential(nn.Linear(width, 256), nn.LayerNorm(256), nn.GELU(),
                                    nn.Dropout(0.35), nn.Linear(256, 128), nn.GELU(),
                                    nn.Dropout(0.2))
        self.classifier = nn.Linear(128, 3)
        self.regressor = nn.Linear(128, 1)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.shared(features)
        return self.classifier(hidden), self.regressor(hidden).squeeze(-1)


def features(batch: dict, bert: nn.Module) -> torch.Tensor:
    with torch.no_grad():
        hidden = bert(input_ids=batch["tokens"],
                      attention_mask=batch["struct"].long(),
                      token_type_ids=batch["segment"]).last_hidden_state.float()
    observed = batch["text_observed"].unsqueeze(-1).float()
    text_mean = (hidden * observed).sum(1) / observed.sum(1).clamp_min(1)
    text = torch.cat([hidden[:, 0], text_mean], dim=-1)
    return torch.cat([text,
                      moments(batch["audio"], batch["audio_observed"]),
                      moments(batch["vision"], batch["vision_observed"])], dim=-1)


def evaluate(split: dict, bert: nn.Module, head: nn.Module, batch_size: int,
             scenario: tuple[str, float, str] | None = None) -> dict:
    head.eval()
    classes, strengths = [], []
    with torch.no_grad():
        for first in range(0, len(split["class"]), batch_size):
            ids = np.arange(first, min(first + batch_size, len(split["class"])))
            batch = slice_batch(split, ids)
            if scenario:
                apply_missing(batch, *scenario)
            class_logits, reg = head(features(batch, bert))
            classes.append(class_logits.argmax(-1).cpu().numpy())
            strengths.append(reg.cpu().numpy())
    y = split["class"].cpu().numpy()
    r = split["strength"].cpu().numpy()
    pred = np.concatenate(classes)
    reg = np.concatenate(strengths)
    return {"accuracy": float(accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, labels=[0, 1, 2], average="macro", zero_division=0)),
            "per_class_f1": f1_score(y, pred, labels=[0, 1, 2], average=None, zero_division=0).tolist(),
            "mae": float(mean_absolute_error(r, reg)),
            "pearson": float(np.corrcoef(r, reg)[0, 1]) if np.std(reg) else 0.0}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
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
    head = FusionHead().to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=3e-4, weight_decay=0.01)
    freq = torch.bincount(train["class"], minlength=3).float()
    weights = (1.0 / freq.sqrt())
    weights /= weights.mean()
    history = []
    best_score, patience = -1e9, 0
    for epoch in range(1, args.epochs + 1):
        head.train()
        order = np.random.permutation(len(train["class"]))
        losses = []
        for first in range(0, len(order), args.batch_size):
            batch = slice_batch(train, order[first:first + args.batch_size])
            augment(batch, rng)
            x = features(batch, bert)
            logits, regression = head(x)
            loss = nn.functional.cross_entropy(logits, batch["class"], weight=weights)
            loss += 0.2 * nn.functional.smooth_l1_loss(regression, batch["strength"])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        metrics = evaluate(valid, bert, head, args.batch_size)
        score = metrics["macro_f1"] + 0.1 * metrics["pearson"] - 0.1 * metrics["mae"]
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "joint_score": score,
               **metrics}
        history.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if score > best_score:
            best_score, patience = score, 0
            torch.save({"head": head.state_dict(), "normalizers": normalizers,
                        "data_version": "aligned_50", "seed": args.seed,
                        "epoch": epoch, "validation": metrics}, OUT / "statistical_fusion_best.pt")
        else:
            patience += 1
            if patience >= 6:
                break
    selected = torch.load(OUT / "statistical_fusion_best.pt", map_location=device, weights_only=False)
    head.load_state_dict(selected["head"])
    scenarios = []
    for modality in MODALITIES:
        for fraction in (0.2, 0.4):
            for position in ("start", "middle", "end"):
                result = evaluate(valid, bert, head, args.batch_size,
                                  (modality, fraction, position))
                scenarios.append({"modality": modality, "fraction": fraction,
                                  "position": position, **result})
    report = {"data_version": "aligned_50", "train_n": len(train["class"]),
              "valid_n": len(valid["class"]), "selection_rule": "macro_f1 + 0.1*pearson - 0.1*mae",
              "seed": args.seed, "selected_epoch": selected["epoch"],
              "selected_validation": selected["validation"],
              "history": history, "missing_scenarios": scenarios}
    (OUT / "statistical_fusion_results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("SELECTED", json.dumps({"epoch": selected["epoch"],
                                 **selected["validation"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
