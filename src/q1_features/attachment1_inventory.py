# 本程序及代码是在人工智能工具辅助下完成的
# 人工智能工具名称：DeepSeek，型号：DeepSeek V4.1 Flash，开发公司：杭州深度求索人工智能基础技术研究有限公司，版本颁布日期：2026-09-10

"""附件 1（100 条原始视频）可复核清单。

以 ``data/attachment1/label-100.xlsx`` 为准，用 ``(video_id, clip_id)`` 精确匹配视频文件，
逐条记录稳定 ID、相对路径、官方英文转写、标签与文件大小，并检查重复、缺失和多余文件。

只读取、不修改任何原始样本或转写。清单写入 ``outputs/q1_features/attachment1_inventory.csv``。
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ATTACHMENT1 = ROOT / "data" / "attachment1"
LABEL_FILE = ATTACHMENT1 / "label-100.xlsx"
CSV_OUTPUT = ROOT / "outputs" / "q1_features" / "attachment1_inventory.csv"

ID_SEPARATOR = "$_$"
VIDEO_SUFFIX = ".mp4"

CSV_FIELDS = [
    "stable_id",
    "video_id",
    "clip_id",
    "relative_path",
    "text",
    "label",
    "annotation",
    "size_bytes",
]


def stable_id(video_id: str, clip_id: str) -> str:
    """与附件 2 一致的稳定 ID 形式：``video_id$_$clip_id``。"""

    return f"{video_id}{ID_SEPARATOR}{clip_id}"


def load_labels() -> list[dict]:
    """读取官方标注表，保留原始文本与数值，不做任何改写。"""

    frame = pd.read_excel(LABEL_FILE)
    expected = ["video_id", "clip_id", "text", "label", "annotation"]
    missing = [c for c in expected if c not in frame.columns]
    if missing:
        raise KeyError(f"{LABEL_FILE.name} 缺少列: {missing}")

    rows = []
    for record in frame.to_dict("records"):
        # clip_id 在表中是整数，文件名是字符串；统一成字符串做精确匹配，不做去零填充等推断。
        clip_raw = record["clip_id"]
        clip_text = str(int(clip_raw)) if isinstance(clip_raw, (int, float)) and float(clip_raw).is_integer() else str(clip_raw)
        rows.append(
            {
                "video_id": str(record["video_id"]),
                "clip_id": clip_text,
                "text": "" if pd.isna(record["text"]) else str(record["text"]),
                "label": float(record["label"]),
                "annotation": "" if pd.isna(record["annotation"]) else str(record["annotation"]),
                "clip_id_raw": clip_raw,
            }
        )
    return rows


def scan_videos() -> list[dict]:
    """扫描附件 1 下的全部视频文件（video_id 目录 / clip_id.mp4）。"""

    videos = []
    for path in sorted(ATTACHMENT1.rglob(f"*{VIDEO_SUFFIX}")):
        rel = path.relative_to(ROOT)
        videos.append(
            {
                "video_id": path.parent.name,
                "clip_id": path.stem,
                "relative_path": str(rel),
                "size_bytes": int(path.stat().st_size),
                "absolute_path": path,
            }
        )
    return videos


def _sign_matches_annotation(label: float, annotation: str) -> bool:
    expected = "Negative" if label < 0 else ("Neutral" if label == 0 else "Positive")
    return annotation == expected


def build_report() -> dict:
    """构建清单行与核对结果。返回结构同时供 CSV 与质量报告使用。"""

    labels = load_labels()
    videos = scan_videos()

    label_index: dict[tuple[str, str], list[dict]] = {}
    for row in labels:
        label_index.setdefault((row["video_id"], row["clip_id"]), []).append(row)
    video_index: dict[tuple[str, str], list[dict]] = {}
    for row in videos:
        video_index.setdefault((row["video_id"], row["clip_id"]), []).append(row)

    rows = []
    unmatched_videos = []
    for key, entries in video_index.items():
        if len(entries) > 1:
            continue
        video = entries[0]
        label = label_index.get(key)
        if not label:
            unmatched_videos.append(video["relative_path"])
            continue
        if len(label) > 1:
            continue
        item = label[0]
        rows.append(
            {
                "stable_id": stable_id(item["video_id"], item["clip_id"]),
                "video_id": item["video_id"],
                "clip_id": item["clip_id"],
                "relative_path": video["relative_path"],
                "text": item["text"],
                "label": item["label"],
                "annotation": item["annotation"],
                "size_bytes": video["size_bytes"],
            }
        )
    rows.sort(key=lambda r: (r["video_id"], int(r["clip_id"])))

    duplicate_labels = sorted(k for k, v in label_index.items() if len(v) > 1)
    duplicate_videos = sorted(k for k, v in video_index.items() if len(v) > 1)
    missing_videos = sorted(set(label_index) - set(video_index))
    extra_videos = sorted(set(video_index) - set(label_index))
    unmatched_label_ids = sorted({r["video_id"] for r in labels} - {v["video_id"] for v in videos})

    # clip_id 前导零检查：若文件名带前导零而表中是整数，精确匹配会失败，需显式记录
    leading_zero_clips = sorted(
        v["clip_id"] for v in videos if re.fullmatch(r"0\d+", v["clip_id"])
    )

    dirs = sorted(p.name for p in ATTACHMENT1.iterdir() if p.is_dir())
    loose_files = sorted(p.name for p in ATTACHMENT1.iterdir() if p.is_file())

    annotation_issues = [
        r["stable_id"] for r in rows if not _sign_matches_annotation(r["label"], r["annotation"])
    ]
    label_out_of_range = [
        r["stable_id"] for r in rows if not (-3.0 <= r["label"] <= 3.0)
    ]

    report = {
        "source": {
            "label_table": str(LABEL_FILE.relative_to(ROOT)),
            "video_root": str(ATTACHMENT1.relative_to(ROOT)),
            "matching_rule": "(video_id, clip_id) 精确匹配；clip_id 统一按字符串比较，不做补零推断",
        },
        "counts": {
            "label_rows": len(labels),
            "video_files": len(videos),
            "matched_rows": len(rows),
            "video_id_dirs": len(dirs),
            "unique_video_id_in_labels": len({r["video_id"] for r in labels}),
            "unique_video_id_in_files": len({v["video_id"] for v in videos}),
            "loose_files_in_root": loose_files,
        },
        "duplicates": {
            "label_duplicate_keys": [list(k) for k in duplicate_labels],
            "video_duplicate_keys": [list(k) for k in duplicate_videos],
        },
        "missing_videos_for_label_rows": [list(k) for k in missing_videos],
        "extra_videos_without_label_rows": [video_index[k][0]["relative_path"] for k in extra_videos],
        "unmatched_video_files": unmatched_videos,
        "label_video_ids_absent_from_disk": unmatched_label_ids,
        "leading_zero_clip_ids": leading_zero_clips,
        "consistency": {
            "exactly_100_label_rows": len(labels) == 100,
            "exactly_100_video_files": len(videos) == 100,
            "matched_equals_label_rows": len(rows) == len(labels),
            "all_ids_unique": len({r["stable_id"] for r in rows}) == len(rows),
            "annotation_mismatch_ids": annotation_issues,
            "label_out_of_range_ids": label_out_of_range,
            "label_range": [min(r["label"] for r in rows), max(r["label"] for r in rows)],
        },
        "rows": rows,
    }
    return report


def write_csv(rows: list[dict]) -> Path:
    CSV_OUTPUT.parent.mkdir(exist_ok=True)
    with CSV_OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in CSV_FIELDS})
    return CSV_OUTPUT


def main() -> None:
    report = build_report()
    path = write_csv(report["rows"])
    summary = {k: v for k, v in report.items() if k != "rows"}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n清单已写入 {path.relative_to(ROOT)}（{len(report['rows'])} 行）")


if __name__ == "__main__":
    main()
