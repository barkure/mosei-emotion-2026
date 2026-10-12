# 本程序及代码是在人工智能工具辅助下完成的
# 人工智能工具名称：DeepSeek，型号：DeepSeek V4.1 Flash，开发公司：杭州深度求索人工智能基础技术研究有限公司，版本颁布日期：2026-09-10

"""核验全部100条视频的解码帧时间、音频时间与共同时间基准。"""

from __future__ import annotations
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
for _p in (str(HERE.parent), str(HERE)):          # src/ 与本文件所在目录
    if _p not in sys.path:
        sys.path.insert(0, _p)


import argparse
import csv
import json
import tempfile
import traceback
from pathlib import Path

from media_io import tool_versions
from media_io import display_path
from media_io import (
    FRAME_TIME_TOLERANCE_SECONDS,
    ROOT,
    SYNC_RATE,
    WAV_RATE,
    decode_wav,
    probe_audio_frames,
    probe_streams,
    probe_video_frames,
    to_float,
)

INVENTORY_CSV = ROOT / "outputs" / "q1_features" / "attachment1_inventory.csv"
OUT_DIR = ROOT / "outputs" / "q1_features" / "attachment1_timebase"
PER_SAMPLE_DIR = OUT_DIR / "per_sample"

TOL_SAMPLES = 1.0 / WAV_RATE + FRAME_TIME_TOLERANCE_SECONDS
AUDIO_FRAME_SECONDS = 1024 / SYNC_RATE

CSV_FIELDS = [
    "stable_id", "video_id", "clip_id", "relative_path", "size_bytes", "status",
    "n_video_frames", "video_first_s", "video_last_s", "video_coverage_end_s", "video_monotonic",
    "video_last_frame_pkt_duration_s",
    "n_audio_frames", "audio_first_s", "audio_last_s", "audio_coverage_end_s", "audio_monotonic",
    "audio_total_nb_samples", "audio_derived_duration_s", "audio_max_residual_s",
    "av_start_offset_s", "av_end_coverage_offset_s", "format_duration_s",
    "wav16k_n_samples", "wav16k_duration_s", "wav16k_minus_derived_s", "wav16k_minus_coverage_s",
    "failed_checks", "anomalies", "failure_reasons",
]


def safe_name(stable_id: str) -> str:
    return stable_id.replace("$_$", "__")


def verify_one(row: dict) -> dict:
    """核验单条样本；任何异常都在内部捕获并记录，不向上抛出。"""

    stable_id = row["stable_id"]
    video = ROOT / row["relative_path"]
    record: dict = {
        "stable_id": stable_id,
        "video_id": row["video_id"],
        "clip_id": row["clip_id"],
        "relative_path": row["relative_path"],
        "size_bytes": int(row["size_bytes"]),
        "official_text_length": len(row["text"]),
    }
    anomalies: list[str] = []
    failure_reasons: list[str] = []
    checks: dict = {}

    try:
        streams = probe_streams(video)
        video_stream = streams["streams"].get("video")
        audio_stream = streams["streams"].get("audio")
        audio_rate = to_float((audio_stream or {}).get("sample_rate"))
        audio_rate = int(audio_rate) if audio_rate else None

        video_frames = probe_video_frames(video, include_arrays=True)
        audio_frames = probe_audio_frames(video, audio_rate, include_arrays=True)
        record["streams"] = streams
        record["video_frames"] = video_frames
        record["audio_frames"] = audio_frames
    except Exception as error:  # noqa: BLE001 - 逐条容错
        record["status"] = "failed"
        record["exception"] = f"{type(error).__name__}: {error}"
        record["traceback"] = traceback.format_exc(limit=3)
        record["anomalies"] = ["probe_failed"]
        record["failure_reasons"] = [record["exception"]]
        record["checks"] = {}
        return record

    v_first = video_frames["first_display_time"]
    v_last = video_frames["last_display_time"]
    v_cov = video_frames["coverage_end_seconds"]
    v_pkt_dur = video_frames["last_frame_pkt_duration_seconds"]
    a_first = audio_frames["first_frame_time"]
    a_last = audio_frames["last_frame_time"]
    a_cov = audio_frames["coverage_end_seconds"]
    a_derived = audio_frames["derived_duration_seconds"]
    a_residual = audio_frames["max_cumulative_sample_time_residual_seconds"]

    av_start = None if (v_first is None or a_first is None) else round(a_first - v_first, 6)
    av_end = None if (v_cov is None or a_cov is None) else round(a_cov - v_cov, 6)

    # --- 临时 16 kHz 解码（用完即删）---
    wav16k = None
    decode_error = None
    try:
        with tempfile.TemporaryDirectory(prefix="tb16k_") as tmp:
            info = decode_wav(video, Path(tmp) / "probe.wav", WAV_RATE)
            wav16k = {
                "n_samples": info["n_samples"],
                "sampling_rate": info["sampling_rate"],
                "duration_seconds": info["duration_seconds"],
                "sha256": info["sha256"],
                "decoded_in_temp_dir_removed_after_use": True,
            }
    except Exception as error:  # noqa: BLE001
        decode_error = f"{type(error).__name__}: {error}"
        anomalies.append("decode_16k_failed")

    wav_dur = wav16k["duration_seconds"] if wav16k else None
    wav_minus_derived = (
        round(wav_dur - a_derived, 6) if (wav_dur is not None and a_derived is not None) else None
    )
    wav_minus_coverage = round(wav_dur - a_cov, 6) if (wav_dur is not None and a_cov is not None) else None

    checks = {
        "video_stream_present": video_stream is not None,
        "audio_stream_present": audio_stream is not None,
        "video_has_frames": video_frames["n_frames"] > 0,
        "video_frame_fields_complete": (
            video_frames["n_frames"] > 0 and video_frames["n_times"] == video_frames["n_frames"]
        ),
        "video_last_frame_duration_available": bool(v_pkt_dur and v_pkt_dur > 0),
        "video_frames_monotonic": video_frames["monotonic_nondecreasing_in_display_order"],
        "audio_has_frames": audio_frames["n_frames"] > 0,
        "audio_frame_fields_complete": audio_frames["frame_fields_complete"],
        "audio_frames_monotonic": audio_frames["monotonic_nondecreasing"],
        "audio_cumulative_sample_residual_within_tolerance": (
            a_residual is not None and a_residual <= FRAME_TIME_TOLERANCE_SECONDS
        ),
        "audio_video_start_offset_zero": (
            av_start is not None and abs(av_start) <= FRAME_TIME_TOLERANCE_SECONDS
        ),
        "temp_16k_decode_ok": wav16k is not None,
        "temp_16k_sample_count_matches_decoded_audio": (
            wav_minus_derived is not None and abs(wav_minus_derived) <= TOL_SAMPLES
        ),
        "temp_16k_duration_matches_audio_coverage_end": (
            wav_minus_coverage is not None and abs(wav_minus_coverage) <= TOL_SAMPLES
        ),
    }
    critical = [
        "video_stream_present",
        "audio_stream_present",
        "video_has_frames",
        "video_frame_fields_complete",
        "video_last_frame_duration_available",
        "video_frames_monotonic",
        "audio_has_frames",
        "audio_frame_fields_complete",
        "audio_frames_monotonic",
        "audio_cumulative_sample_residual_within_tolerance",
        "audio_video_start_offset_zero",
        "temp_16k_decode_ok",
        "temp_16k_sample_count_matches_decoded_audio",
        "temp_16k_duration_matches_audio_coverage_end",
    ]
    failed_checks = [name for name, ok in checks.items() if not ok]

    # --- 异常标记（含不作为失败判据、但必须报出的项）---
    if not checks["video_frames_monotonic"]:
        anomalies.append("video_frames_not_monotonic")
    if not checks["audio_frames_monotonic"]:
        anomalies.append("audio_frames_not_monotonic")
    if not checks["audio_cumulative_sample_residual_within_tolerance"]:
        anomalies.append("audio_cumulative_residual_over_tolerance")
    if not checks["video_last_frame_duration_available"]:
        anomalies.append("video_missing_last_frame_duration")
    if not checks["video_frame_fields_complete"] or not checks["audio_frame_fields_complete"]:
        anomalies.append("frame_fields_incomplete")
    if video_frames["n_frames"] == 0:
        anomalies.append("empty_video_frames")
    if audio_frames["n_frames"] == 0:
        anomalies.append("empty_audio_frames")
    if video_stream is None:
        anomalies.append("missing_video_stream")
    if audio_stream is None:
        anomalies.append("missing_audio_stream")
    if not checks["audio_video_start_offset_zero"]:
        anomalies.append("audio_video_start_offset_nonzero")
    if av_end is not None and abs(av_end) > AUDIO_FRAME_SECONDS:
        anomalies.append("audio_video_end_offset_over_audio_frame")
    if wav_minus_coverage is not None and abs(wav_minus_coverage) > TOL_SAMPLES:
        anomalies.append("decode_16k_duration_mismatch")

    status = "verified" if not failed_checks else "failed"
    if failed_checks:
        failure_reasons = [f"未通过的检查: {', '.join(failed_checks)}"]
        if decode_error:
            failure_reasons.append(f"16 kHz 解码失败: {decode_error}")

    record.update(
        {
            "checks": checks,
            "critical_checks": critical,
            "failed_checks": failed_checks,
            "status": status,
            "failure_reasons": failure_reasons,
            "anomalies": sorted(set(anomalies)),
            "av_offset": {
                "start_seconds": av_start,
                "end_coverage_seconds": av_end,
                "note": (
                    "start 用两路解码首帧时间之差；end_coverage 用各自“末帧起点 + 该帧时长”之差。"
                    "两者都是**观测值**，非失败判据。"
                ),
            },
            "temp_16k_decode": wav16k,
            "derived": {
                "wav16k_minus_derived_seconds": wav_minus_derived,
                "wav16k_minus_audio_coverage_end_seconds": wav_minus_coverage,
                "tolerance_seconds": TOL_SAMPLES,
            },
            "verification_scope": (
                "verified 仅表示**解码帧级**核验通过（字段完整、帧时间单调、音频帧时间与累计样本时间"
                f"残差 ≤ {FRAME_TIME_TOLERANCE_SECONDS}s、16 kHz 临时解码样本数与时长一致）。"
                "**不表示**重采样后的样本级绝对延迟已标定，也不涉及词边界、语种或官方转写。"
            ),
        }
    )
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="附件 1 全部 100 条视频的解码帧级音画时间基准核验")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--limit", type=int, default=0, help="仅处理前 N 条（调试用，0 表示全部）")
    args = parser.parse_args()

    out_dir = args.out_dir
    per_sample_dir = out_dir / "per_sample"
    per_sample_dir.mkdir(parents=True, exist_ok=True)

    with INVENTORY_CSV.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if args.limit:
        rows = rows[: args.limit]
    print(f"清单 {len(rows)} 条，开始逐条核验……")

    records = []
    for index, row in enumerate(rows, start=1):
        record = verify_one(row)
        records.append(record)
        (per_sample_dir / f"{safe_name(record['stable_id'])}.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        if index % 20 == 0 or index == len(rows):
            failed = sum(1 for r in records if r["status"] != "verified")
            print(f"  已处理 {index}/{len(rows)}（未通过 {failed}）")

    # --- 汇总 CSV（恰好 len(rows) 行）---
    csv_path = out_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for r in records:
            vf = r.get("video_frames") or {}
            af = r.get("audio_frames") or {}
            wav = r.get("temp_16k_decode") or {}
            writer.writerow(
                {
                    "stable_id": r["stable_id"],
                    "video_id": r["video_id"],
                    "clip_id": r["clip_id"],
                    "relative_path": r["relative_path"],
                    "size_bytes": r["size_bytes"],
                    "status": r["status"],
                    "n_video_frames": vf.get("n_frames"),
                    "video_first_s": vf.get("first_display_time"),
                    "video_last_s": vf.get("last_display_time"),
                    "video_coverage_end_s": vf.get("coverage_end_seconds"),
                    "video_monotonic": vf.get("monotonic_nondecreasing_in_display_order"),
                    "video_last_frame_pkt_duration_s": vf.get("last_frame_pkt_duration_seconds"),
                    "n_audio_frames": af.get("n_frames"),
                    "audio_first_s": af.get("first_frame_time"),
                    "audio_last_s": af.get("last_frame_time"),
                    "audio_coverage_end_s": af.get("coverage_end_seconds"),
                    "audio_monotonic": af.get("monotonic_nondecreasing"),
                    "audio_total_nb_samples": af.get("total_nb_samples"),
                    "audio_derived_duration_s": af.get("derived_duration_seconds"),
                    "audio_max_residual_s": af.get("max_cumulative_sample_time_residual_seconds"),
                    "av_start_offset_s": (r.get("av_offset") or {}).get("start_seconds"),
                    "av_end_coverage_offset_s": (r.get("av_offset") or {}).get("end_coverage_seconds"),
                    "format_duration_s": to_float((r.get("streams") or {}).get("format", {}).get("duration")),
                    "wav16k_n_samples": wav.get("n_samples"),
                    "wav16k_duration_s": wav.get("duration_seconds"),
                    "wav16k_minus_derived_s": (r.get("derived") or {}).get("wav16k_minus_derived_seconds"),
                    "wav16k_minus_coverage_s": (r.get("derived") or {}).get("wav16k_minus_audio_coverage_end_seconds"),
                    "failed_checks": ";".join(r.get("failed_checks") or []),
                    "anomalies": ";".join(r.get("anomalies") or []),
                    "failure_reasons": " | ".join(r.get("failure_reasons") or []),
                }
            )

    # --- 异常 ID 清单 ---
    anomaly_index: dict[str, list[str]] = {}
    for r in records:
        for kind in r.get("anomalies") or []:
            anomaly_index.setdefault(kind, []).append(r["stable_id"])
    failed_ids = [r["stable_id"] for r in records if r["status"] != "verified"]
    anomaly_payload = {
        "n_samples": len(records),
        "n_verified": sum(1 for r in records if r["status"] == "verified"),
        "n_failed": len(failed_ids),
        "failed_ids": failed_ids,
        "anomaly_counts": {k: len(v) for k, v in sorted(anomaly_index.items())},
        "anomaly_ids": {k: sorted(v) for k, v in sorted(anomaly_index.items())},
    }
    (out_dir / "anomaly_ids.json").write_text(
        json.dumps(anomaly_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    def _rng(values, digits=6):
        # format.duration 等字段在 ffprobe JSON 里是字符串，统一转成 float 再统计
        clean = [v for v in (to_float(x) for x in values) if v is not None]
        return [round(min(clean), digits), round(max(clean), digits)] if clean else None

    residual_pairs = [
        (r["stable_id"], (r.get("audio_frames") or {}).get("max_cumulative_sample_time_residual_seconds"))
        for r in records
    ]
    residuals = [value for _, value in residual_pairs if value is not None]
    worst = None
    if residuals:
        worst = max(
            (pair for pair in residual_pairs if pair[1] is not None),
            key=lambda pair: pair[1],
        )
        worst = {"stable_id": worst[0], "max_residual_seconds": round(worst[1], 9)}

    summary = {
        "purpose": "附件 1 全部 100 条视频的解码帧级音画时间基准核验",
        "tool_versions": tool_versions(),
        "scope": (
            "只做解码帧级核验：字段完整性、帧时间单调性、音频帧时间与累计样本时间的残差、音画起点/终点偏移，"
            "以及临时 16 kHz 解码的样本数与时长。不读取附件 2—4，不训练模型，不做词级对齐或情感特征提取。"
        ),
        "verified_definition": (
            "verified 仅指上述解码帧级检查全部通过；**不宣称**重采样后的样本级绝对延迟已标定。"
        ),
        "tolerances": {
            "frame_time_tolerance_seconds": FRAME_TIME_TOLERANCE_SECONDS,
            "wav16k_tolerance_seconds": TOL_SAMPLES,
            "audio_frame_seconds": round(AUDIO_FRAME_SECONDS, 6),
            "note": "音频帧时间残差容差 = ffprobe 时间戳的 6 位小数分辨率；16 kHz 时长容差 = 1 个样本 + 该分辨率",
        },
        "counts": {
            "n_samples": len(records),
            "n_verified": sum(1 for r in records if r["status"] == "verified"),
            "n_failed": len(failed_ids),
            "n_with_anomalies": sum(1 for r in records if r.get("anomalies")),
            "total_video_frames": sum((r.get("video_frames") or {}).get("n_frames", 0) for r in records),
            "total_audio_frames": sum((r.get("audio_frames") or {}).get("n_frames", 0) for r in records),
        },
        "ranges": {
            "av_start_offset_seconds": _rng([(r.get("av_offset") or {}).get("start_seconds") for r in records]),
            "av_end_coverage_offset_seconds": _rng([(r.get("av_offset") or {}).get("end_coverage_seconds") for r in records]),
            "audio_max_residual_seconds": _rng(residuals, digits=9),
            "video_coverage_end_seconds": _rng([(r.get("video_frames") or {}).get("coverage_end_seconds") for r in records]),
            "audio_coverage_end_seconds": _rng([(r.get("audio_frames") or {}).get("coverage_end_seconds") for r in records]),
            "video_frame_count": _rng([(r.get("video_frames") or {}).get("n_frames") for r in records]),
            "audio_frame_count": _rng([(r.get("audio_frames") or {}).get("n_frames") for r in records]),
            "format_duration_seconds": _rng([(r.get("streams") or {}).get("format", {}).get("duration") for r in records]),
        },
        "largest_audio_residual": worst,
        "anomaly_counts": anomaly_payload["anomaly_counts"],
        "failed_ids": failed_ids,
        "outputs": {
            "per_sample_dir": display_path(per_sample_dir),
            "summary_csv": display_path(csv_path),
            "anomaly_ids": display_path(out_dir / "anomaly_ids.json"),
        },
    }
    (out_dir / "timebase_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"\n输出目录: {display_path(out_dir)}")
    print(f"  通过 {summary['counts']['n_verified']} / 未通过 {summary['counts']['n_failed']} / 共 {len(records)}")
    print(f"  音画起点偏移范围: {summary['ranges']['av_start_offset_seconds']}")
    print(f"  音画终点(覆盖)偏移范围: {summary['ranges']['av_end_coverage_offset_seconds']}")
    print(f"  音频累计样本时间最大残差: {summary['largest_audio_residual']}")
    print(f"  异常类型计数: {summary['anomaly_counts']}")
    if failed_ids:
        raise SystemExit("Media timebase verification failed; inspect per-sample reports.")


if __name__ == "__main__":
    main()
