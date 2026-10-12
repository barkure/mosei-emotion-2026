# 本程序及代码是在人工智能工具辅助下完成的
# 人工智能工具名称：DeepSeek，型号：DeepSeek V4.1 Flash，开发公司：杭州深度求索人工智能基础技术研究有限公司，版本颁布日期：2026-09-10

"""Summarize supplied MOSEI splits and the two specialist inputs."""
from __future__ import annotations
import json
import pickle
from collections import Counter
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def summary(value):
    if isinstance(value, np.ndarray):
        result = {"shape": list(value.shape), "dtype": str(value.dtype)}
        if value.dtype.kind in "fiu" and value.size:
            result["finite_fraction"] = float(np.isfinite(value).mean())
            if value.ndim >= 2:
                result["all_zero_row_fraction"] = float(np.all(value == 0, axis=-1).mean())
        return result
    if isinstance(value, (list, tuple)):
        return {"length": len(value), "first_type": type(value[0]).__name__ if value else None}
    return {"type": type(value).__name__, "preview": str(value)[:80]}


def main():
    with (ROOT / "data/attachment2/aligned_50.pkl").open("rb") as f:
        data = pickle.load(f)
    report = {"splits": {}, "specialist": {}}
    for split, fields in data.items():
        report["splits"][split] = {key: summary(value) for key, value in fields.items()}
        if "classification_labels" in fields:
            arr = np.asarray(fields["classification_labels"]).reshape(-1)
            report["splits"][split]["class_counts"] = dict(Counter(map(str, arr)))
        if "regression_labels" in fields:
            arr = np.asarray(fields["regression_labels"]).reshape(-1)
            report["splits"][split]["regression_range"] = [float(np.nanmin(arr)), float(np.nanmax(arr))]
    specialist_groups = {
        "attachment3": "attachment3/aligned",
        "attachment4": "attachment4/aligned",
        "attachment3_unaligned": "attachment3/unaligned",
        "attachment4_unaligned": "attachment4/unaligned",
    }
    for group, relative in specialist_groups.items():
        paths = sorted((ROOT / "data" / relative).glob("*.pkl"))
        report["specialist"][group] = {"files": len(paths)}
        if paths:
            with paths[0].open("rb") as f:
                sample = pickle.load(f)
            fields = sample.get("test", sample)
            report["specialist"][group]["sample_fields"] = {key: summary(value) for key, value in fields.items()}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    (ROOT / "outputs").mkdir(exist_ok=True)
    (ROOT / "outputs/data_audit/data_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
