# 本程序及代码是在人工智能工具辅助下完成的
# 人工智能工具名称：DeepSeek，型号：DeepSeek V4.1 Flash，开发公司：杭州深度求索人工智能基础技术研究有限公司，版本颁布日期：2026-09-10

"""Shared IO and native interval aggregation for the final Q1 pipeline."""
import csv
import hashlib
import numpy as np
PITCH_DIM=14

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return [{k.strip(): v.strip() for k, v in row.items()}
                for row in csv.DictReader(f, skipinitialspace=True)]


def write_rows(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def overlap_sources(start, end, left, right):
    overlap = np.maximum(0., np.minimum(right, end) - np.maximum(left, start))
    indices = np.flatnonzero(overlap > 1e-9)
    lengths = overlap[indices]
    return indices, lengths, lengths / lengths.sum() if len(lengths) else lengths


def pooled_audio(features, pitch_mask, indices, weights):
    out = np.zeros(features.shape[1], dtype=np.float64)
    if len(indices):
        out = weights @ features[indices].astype(np.float64)
        voiced = pitch_mask[indices]
        out[PITCH_DIM] = (weights[voiced] @ features[indices[voiced], PITCH_DIM]
                          / weights[voiced].sum()) if voiced.any() else 0.
    return out


def csr(rows):
    return (np.r_[0, np.cumsum([len(r) for r in rows])].astype(np.int32),
            np.concatenate(rows).astype(np.int32) if rows else np.array([], np.int32))
