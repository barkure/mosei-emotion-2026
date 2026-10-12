# 本程序及代码是在人工智能工具辅助下完成的
# 人工智能工具名称：DeepSeek，型号：DeepSeek V4.1 Flash，开发公司：杭州深度求索人工智能基础技术研究有限公司，版本颁布日期：2026-09-10



"""附件 2 对齐版 / 未对齐版逐样本审计。

只读取附件 2 的 ``train`` 与 ``valid``（题面禁止用 ``test`` 做分析，也不读附件 3、4）。
结果写入**独立的新文件** ``outputs/data_audit/attachment2_audit.json``，
不覆盖也不依赖既有的 ``outputs/data_audit/data_audit.json``。

检查项：样本数、ID 唯一性与跨划分重复、字段形状与 dtype、长度范围、
非有限值、标签取值与“分类=强度符号”的对应关系，以及结构有效区间内的零行比例。

对未对齐版视觉同时报告**原始声明长度**与**实际使用的结构长度**，
越界非零行一律以原始声明长度为基准统计，并单独报告被实际掩码保留下来的行数。
"""

from __future__ import annotations
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
for _p in (str(HERE.parent), str(HERE)):          # src/ 与本文件所在目录
    if _p not in sys.path:
        sys.path.insert(0, _p)


import json
from collections import Counter
from pathlib import Path

import numpy as np

from data_io import (
    CLS_ID,
    ROOT,
    SEP_ID,
    SPECIAL_TOKEN_IDS,
    Attachment2,
    SplitData,
)

AUDITED_SPLITS = ("train", "valid")
OUTPUT = ROOT / "outputs" / "data_audit" / "attachment2_audit.json"


def _rng(values: np.ndarray) -> dict:
    values = np.asarray(values)
    return {"min": float(values.min()), "max": float(values.max()), "mean": float(values.mean())}


def _nonfinite(signal: np.ndarray) -> int:
    return int((~np.isfinite(np.asarray(signal, dtype=np.float64))).sum())


def audit_split(data: SplitData, version_other: SplitData | None) -> dict:
    """审计单个 (version, split)。"""

    report: dict = {
        "version": data.version,
        "split": data.split,
        "n_samples": len(data),
        "meta": data.meta,
    }

    # ---- ID ----
    ids = [str(x) for x in data.sample_id]
    report["id"] = {
        "n_unique": len(set(ids)),
        "duplicates": int(len(ids) - len(set(ids))),
        "all_match_video_clip_format": bool(
            all(str(v) and str(c) for v, c in zip(data.video_id, data.clip_id))
        ),
        "n_unique_video_id": int(len(set(map(str, data.video_id)))),
    }
    if version_other is not None:
        other_ids = [str(x) for x in version_other.sample_id]
        report["id"]["aligned_unaligned_same_order"] = ids == other_ids
        report["id"]["aligned_unaligned_same_set"] = set(ids) == set(other_ids)

    # ---- 形状与 dtype ----
    report["shapes"] = {
        "text": list(data.text.shape),
        "text_tokens": list(data.text_tokens.shape),
        "text_segment": list(data.text_segment.shape),
        "audio": list(data.audio.shape),
        "vision": list(data.vision.shape),
        "text_struct_mask": list(data.text_struct_mask.shape),
        "audio_struct_mask": list(data.audio_struct_mask.shape),
        "vision_struct_mask": list(data.vision_struct_mask.shape),
    }
    report["dtypes"] = {
        "text": str(data.text.dtype),
        "audio": str(data.audio.dtype),
        "vision": str(data.vision.dtype),
        "text_tokens": str(data.text_tokens.dtype),
        "labels_classification": str(data.classification_labels.dtype),
        "labels_regression": str(data.regression_labels.dtype),
    }

    # ---- 非有限值 ----
    report["nonfinite"] = {
        "text": _nonfinite(data.text),
        "audio": _nonfinite(data.audio),
        "vision": _nonfinite(data.vision),
        "regression_labels": _nonfinite(data.regression_labels),
    }

    # ---- 文本结构 ----
    lengths = data.text_lengths
    positions = np.arange(data.text_tokens.shape[1])[None, :]
    outside_nonzero = (data.text_tokens != 0) & ~data.text_struct_mask
    text_vectors = np.asarray(data.text != 0).any(axis=-1)
    report["text"] = {
        "attn_len_range": _rng(lengths),
        "attn_len_histogram": {str(k): int(v) for k, v in sorted(Counter(lengths.tolist()).items())},
        "mask_is_contiguous_prefix": bool((data.text_struct_mask == (positions < lengths[:, None])).all()),
        "first_token_all_cls": bool((data.text_tokens[:, 0] == CLS_ID).all()),
        "last_valid_token_all_sep": bool(
            (data.text_tokens[np.arange(len(lengths)), lengths - 1] == SEP_ID).all()
        ),
        "segment_ids_all_zero": bool((data.text_segment[data.text_struct_mask] == 0).all()),
        "nonzero_tokens_outside_mask": int(outside_nonzero.sum()),
        "special_tokens_inside_mask": {
            "pad": int((data.text_tokens[data.text_struct_mask] == 0).sum()),
            "cls": int((data.text_tokens[data.text_struct_mask] == CLS_ID).sum()),
            "sep": int((data.text_tokens[data.text_struct_mask] == SEP_ID).sum()),
            "other_special_like": int(
                np.isin(data.text_tokens[data.text_struct_mask], [100, 103]).sum()
            ),
        },
        "content_positions": int(data.text_content_mask.sum()),
        "special_positions": int(data.text_struct_mask.sum() - data.text_content_mask.sum()),
        # text_observed_mask 只覆盖内容词元
        "observed_mask_equals_content_mask": bool(
            np.array_equal(data.text_observed_mask, data.text_content_mask)
        ),
        "observed_positions": int(data.text_observed_mask.sum()),
        "undetectable_text_occlusion": (
            "字段无法识别“词元与注意力掩码均未改变的文本遮挡”；"
            "text_observed_mask 只表示该位置是有效内容词元，不代表文本内容完整可用"
        ),
        # 文本 768 维向量在 padding 位置也可能非零，因此不能用全零行判断长度
        "samples_with_nonzero_text_rows_beyond_mask": int(
            (text_vectors & ~data.text_struct_mask).any(axis=1).sum()
        ),
        "text_768d_zero_rows_inside_mask": int((~text_vectors & data.text_struct_mask).sum()),
    }

    # ---- 音频 / 视觉 ----
    modality = {}
    for name in ("audio", "vision"):
        signal = getattr(data, name)
        struct = getattr(data, f"{name}_struct_mask")
        observed = getattr(data, f"{name}_observed_mask")
        beyond = getattr(data, f"{name}_nonzero_beyond_declared")
        rescued = getattr(data, f"{name}_nonzero_rescued")
        declared = getattr(data, f"{name}_declared_lengths")
        struct_len = getattr(data, f"{name}_struct_lengths")
        lengths_field = getattr(data, f"{name}_lengths")
        nz_rows = np.asarray(signal != 0).any(axis=-1).sum(axis=1)
        entry = {
            "positions": int(signal.shape[1]),
            "dim": int(signal.shape[2]),
            # 声明长度（原始字段；aligned 下等于文本注意力长度）
            "declared_length_range": _rng(declared),
            # 实际使用的结构长度
            "struct_len_range": _rng(struct_len),
            "struct_length_extended_vs_declared": int((struct_len > declared).sum()),
            "struct_extra_positions_total": int((struct_len - declared).sum()),
            "struct_positions": int(struct.sum()),
            "observed_positions": int(observed.sum()),
            "zero_row_ratio_in_struct": float(1.0 - observed.sum() / max(1, struct.sum())),
            "zero_row_ratio_whole_tensor": float(
                1.0 - np.asarray(signal != 0).any(axis=-1).sum() / signal.shape[0] / signal.shape[1]
            ),
            # 越界统计：一律以原始声明长度为基准，不因结构扩展而清零
            "nonzero_rows_beyond_declared": {
                "basis": "original declared length field",
                "n_samples": int((beyond > 0).sum()),
                "max_rows": int(beyond.max()),
                "total_rows": int(beyond.sum()),
            },
            # 其中被实际结构掩码保留下来的越界非零行
            "nonzero_rows_rescued_by_struct": {
                "n_samples": int((rescued > 0).sum()),
                "max_rows": int(rescued.max()),
                "total_rows": int(rescued.sum()),
            },
            "declared_length_field": None,
        }
        if lengths_field is not None:
            entry["declared_length_field"] = {
                "name": f"{name}_lengths",
                "range": _rng(lengths_field),
                "equals_nonzero_row_count": float((nz_rows == lengths_field).mean()),
                "nonzero_rows_less_than_declared": int((nz_rows < lengths_field).sum()),
                "nonzero_rows_greater_than_declared": int((nz_rows > lengths_field).sum()),
                "zero_rows_inside_declared": int(
                    np.array(
                        [
                            int(((signal[i, : lengths_field[i]] == 0).all(axis=-1)).sum())
                            for i in range(signal.shape[0])
                        ]
                    ).sum()
                ),
                "samples_with_zero_rows_inside_declared": int(
                    sum(
                        1
                        for i in range(signal.shape[0])
                        if ((signal[i, : lengths_field[i]] == 0).all(axis=-1)).any()
                    )
                ),
            }
        modality[name] = entry
    report["modalities"] = modality

    # 对齐版专属：特殊词元位置（[CLS] / [SEP]）是否为全零行
    if data.version == "aligned":
        idx = np.arange(len(lengths))
        audio_zero = ~np.asarray(data.audio != 0).any(axis=-1)
        vision_zero = ~np.asarray(data.vision != 0).any(axis=-1)
        report["aligned_special_positions"] = {
            "position0_all_zero_audio": float(audio_zero[:, 0].mean()),
            "position0_all_zero_vision": float(vision_zero[:, 0].mean()),
            "position_last_valid_all_zero_audio": float(audio_zero[idx, lengths - 1].mean()),
            "position_last_valid_all_zero_vision": float(vision_zero[idx, lengths - 1].mean()),
        }

    # ---- 标签 ----
    cls = data.classification_labels
    reg = data.regression_labels
    # 精确取值检查：不做整数转换，避免把非整数取值误判为合法类别
    cls_exact_values = sorted(set(np.asarray(cls, dtype=np.float64).tolist()))
    class_counts = Counter(np.asarray(cls, dtype=np.float64).tolist())
    mapped = np.sign(reg).astype(int) + 1
    report["labels"] = {
        "classification_exact_values": [float(v) for v in cls_exact_values],
        "classification_exact_subset_of_0_1_2": bool(set(cls_exact_values) <= {0.0, 1.0, 2.0}),
        "classification_counts": {str(k): int(v) for k, v in sorted(class_counts.items())},
        "classification_all_integral": bool(np.array_equal(np.asarray(cls, dtype=np.float64), np.round(np.asarray(cls, dtype=np.float64)))),
        "regression_range": [float(reg.min()), float(reg.max())],
        "regression_within_minus3_3": bool((reg >= -3).all() and (reg <= 3).all()),
        "regression_zero_count": int((reg == 0).sum()),
        "consistency_class_equals_sign_of_intensity": float((mapped == cls).mean()),
        "consistency_violations": int((mapped != cls).sum()),
        "negative_intensity_with_neutral_class": int(((reg < 0) & (cls == 1)).sum()),
        "positive_intensity_with_neutral_class": int(((reg > 0) & (cls == 1)).sum()),
    }

    report["summary"] = data.modality_summary()
    return report


def main() -> None:
    report: dict = {
        "scope": {
            "splits_audited": list(AUDITED_SPLITS),
            "note": (
                "仅审计附件 2 的 train 与 valid；不使用 test 划分，不读取附件 3、4 的输入分布。"
            ),
            "mask_contract": {
                "text_observed": "content tokens only (= text_content_mask)",
                "audio_struct": "position < declared length (audio_lengths for unaligned)",
                "vision_struct": (
                    "unaligned: position < min(500, max(vision_lengths, last_nonzero+1)); "
                    "conservative reading rule from attachment-2 train/valid only, "
                    "does not claim to resolve the original semantics of vision_lengths"
                ),
                "beyond_declared_basis": "original declared length field (not expanded struct length)",
            },
        },
        "versions": {},
        "cross_version": {},
    }

    for version in ("aligned", "unaligned"):
        reader = Attachment2(version)
        data = {name: reader.split(name) for name in AUDITED_SPLITS}
        report["versions"][version] = {
            "file": reader.path.name,
            "file_bytes": int(reader.path.stat().st_size),
            "splits": {name: audit_split(data[name], None) for name in AUDITED_SPLITS},
        }
        # 同版本内跨划分 ID 重复
        a = set(map(str, data["train"].sample_id))
        b = set(map(str, data["valid"].sample_id))
        report["versions"][version]["train_valid_id_overlap"] = len(a & b)
        del reader, data

    # 跨版本一致性
    aligned = Attachment2("aligned")
    unaligned = Attachment2("unaligned")
    cross = {}
    for name in AUDITED_SPLITS:
        left = aligned.split(name)
        right = unaligned.split(name)
        cross[name] = {
            "n_equal": len(left) == len(right),
            "ids_same_order": [str(x) for x in left.sample_id] == [str(x) for x in right.sample_id],
            "text_bert_identical": bool(
                np.array_equal(left.text_tokens, right.text_tokens)
                and np.array_equal(left.text_struct_mask, right.text_struct_mask)
            ),
            "text_vectors_identical": bool(np.array_equal(left.text, right.text)),
            "labels_identical": bool(
                np.array_equal(left.classification_labels, right.classification_labels)
                and np.array_equal(left.regression_labels, right.regression_labels)
            ),
        }
    report["cross_version"] = cross

    OUTPUT.parent.mkdir(exist_ok=True)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
