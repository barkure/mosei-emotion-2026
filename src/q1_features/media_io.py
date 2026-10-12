# 本程序及代码是在人工智能工具辅助下完成的
# 人工智能工具名称：DeepSeek，型号：DeepSeek V4.1 Flash，开发公司：杭州深度求索人工智能基础技术研究有限公司，版本颁布日期：2026-09-10

"""媒体解码、时间戳与来源路径工具。"""

from __future__ import annotations

import hashlib

import json

import statistics

import subprocess
import tempfile

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

FFMPEG = "ffmpeg"

FFPROBE = "ffprobe"

WAV_RATE = 16000

SYNC_RATE = 44100

AAC_FRAME_SAMPLES = 1024

FRAME_TIME_TOLERANCE_SECONDS = 2e-6

def run(command: list[str]) -> subprocess.CompletedProcess:
    # Capture to disk and require complete JSON before accepting FFprobe output.
    # Some Windows FFprobe builds occasionally return a truncated document.
    expects_json = any(command[i] in ('-of', '-print_format') and
                       command[i + 1] == 'json' for i in range(len(command) - 1))
    for attempt in range(3):
        with tempfile.TemporaryFile() as output:
            result = subprocess.run(command, stdout=output, stderr=subprocess.PIPE,
                                    timeout=600, check=True)
            output.seek(0)
            text = output.read().decode("utf-8")
        if expects_json:
            try:
                json.loads(text)
            except json.JSONDecodeError:
                if attempt == 2:
                    raise
                continue
        return subprocess.CompletedProcess(command, result.returncode, text,
                                           result.stderr.decode("utf-8", errors="replace"))
    raise RuntimeError("Media probe did not produce a complete result")


def to_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()

def probe_streams(video: Path) -> dict:
    result = run([FFPROBE, "-v", "error", "-show_streams", "-show_format", "-of", "json", str(video)])
    payload = json.loads(result.stdout or "{}")
    out = {"format": {}, "streams": {}}
    fmt = payload.get("format", {})
    out["format"] = {
        "start_time": fmt.get("start_time"),
        "duration": fmt.get("duration"),
        "nb_streams": fmt.get("nb_streams"),
    }
    for stream in payload.get("streams", []):
        kind = stream.get("codec_type")
        if kind in ("video", "audio"):
            out["streams"][kind] = {
                key: stream.get(key)
                for key in (
                    "codec_name", "time_base", "start_time", "start_pts", "duration",
                    "nb_frames", "r_frame_rate", "avg_frame_rate", "sample_rate", "channels",
                    "width", "height",
                )
            }
    return out

def probe_video_frames(video: Path, include_arrays: bool = False) -> dict:
    result = run(
        [
            FFPROBE, "-v", "error", "-select_streams", "v:0",
            "-show_entries", "frame=best_effort_timestamp_time,pts_time,pkt_dts_time,duration_time,pkt_duration_time,key_frame",
            "-of", "json", str(video),
        ]
    )
    frames = (json.loads(result.stdout or "{}").get("frames") or [])
    times = [t for t in (to_float(f.get("best_effort_timestamp_time")) for f in frames) if t is not None]
    deltas = [round(times[i] - times[i - 1], 6) for i in range(1, len(times))]
    decreases = [i for i in range(1, len(times)) if times[i] < times[i - 1]]
    last_duration = to_float(frames[-1].get("duration_time", frames[-1].get("pkt_duration_time"))) if frames else None
    coverage_end = (
        round(times[-1] + last_duration, 6)
        if len(times) == len(frames) and last_duration is not None and last_duration > 0
        else None
    )
    return {
        "n_frames": len(frames),
        "n_times": len(times),
        "first_display_time": round(times[0], 6) if times else None,
        "last_display_time": round(times[-1], 6) if times else None,
        "span_seconds": round(times[-1] - times[0], 6) if times else None,
        "monotonic_nondecreasing_in_display_order": bool(times) and len(decreases) == 0,
        "n_decreases": len(decreases),
        "median_frame_interval": round(statistics.median(deltas), 6) if deltas else None,
        "last_frame_interval": deltas[-1] if deltas else None,
        "last_frame_pkt_duration_seconds": last_duration,
        "coverage_end_method": "last_frame_pts_plus_duration" if coverage_end is not None else "unverified_missing_last_frame_duration",
        "coverage_end_seconds": coverage_end,
        "n_key_frames": sum(1 for f in frames if f.get("key_frame") == 1),
        **(
            {
                "display_times": [round(v, 6) for v in times],
                "pkt_durations": [to_float(f.get("duration_time", f.get("pkt_duration_time"))) for f in frames],
            }
            if include_arrays
            else {}
        ),
    }

def probe_audio_frames(video: Path, stream_sample_rate: int | None, include_arrays: bool = False) -> dict:
    result = run(
        [
            FFPROBE, "-v", "error", "-select_streams", "a:0",
            "-show_entries", "frame=best_effort_timestamp_time,pts_time,nb_samples,sample_rate",
            "-of", "json", str(video),
        ]
    )
    frames = (json.loads(result.stdout or "{}").get("frames") or [])
    times = [t for t in (to_float(f.get("best_effort_timestamp_time")) for f in frames) if t is not None]
    samples = [int(f["nb_samples"]) for f in frames if f.get("nb_samples") is not None]
    fields_complete = bool(frames) and len(times) == len(frames) and len(samples) == len(frames) and all(n > 0 for n in samples)
    # 注意：ffprobe 的音频 frame 条目不返回 sample_rate，必须以流信息为准
    rates = [str(stream_sample_rate)] if stream_sample_rate else []
    deltas = [round(times[i] - times[i - 1], 6) for i in range(1, len(times))]
    decreases = [i for i in range(1, len(times)) if times[i] < times[i - 1]]
    total_samples = sum(samples)
    residuals = []
    if fields_complete and stream_sample_rate:
        prior_samples = 0
        for timestamp, n_samples in zip(times, samples):
            residuals.append(timestamp - (times[0] + prior_samples / stream_sample_rate))
            prior_samples += n_samples
    max_residual = max((abs(value) for value in residuals), default=None)
    return {
        "n_frames": len(frames),
        "n_times": len(times),
        "first_frame_time": round(times[0], 6) if times else None,
        "last_frame_time": round(times[-1], 6) if times else None,
        "span_seconds": round(times[-1] - times[0], 6) if times else None,
        "monotonic_nondecreasing": bool(times) and len(decreases) == 0,
        "n_decreases": len(decreases),
        "median_frame_interval": round(statistics.median(deltas), 6) if deltas else None,
        "total_nb_samples": total_samples,
        "frame_fields_complete": fields_complete,
        "max_cumulative_sample_time_residual_seconds": round(max_residual, 9) if max_residual is not None else None,
        "cumulative_sample_time_tolerance_seconds": FRAME_TIME_TOLERANCE_SECONDS,
        "sample_rates": rates,
        "derived_duration_seconds": round(total_samples / stream_sample_rate, 6) if fields_complete and stream_sample_rate else None,
        "first_frame_nb_samples": samples[0] if samples else None,
        "full_frame_nb_samples": AAC_FRAME_SAMPLES,
        "last_frame_nb_samples": samples[-1] if samples else None,
        # 末帧“覆盖终点”= 末帧起点 + 末帧样本数 / 采样率
        "coverage_end_seconds": (
            round(times[-1] + samples[-1] / stream_sample_rate, 6) if fields_complete and stream_sample_rate else None
        ),
        **(
            {
                "frame_times": [round(v, 6) for v in times],
                "nb_samples": samples,
                "cumulative_sample_time_residuals": [round(v, 9) for v in residuals],
            }
            if include_arrays
            else {}
        ),
    }

def decode_wav(video: Path, target: Path, rate: int) -> dict:
    target.parent.mkdir(parents=True, exist_ok=True)
    result = run(
        [
            FFMPEG, "-v", "error", "-y", "-i", str(video),
            "-vn", "-ac", "1", "-ar", str(rate), "-c:a", "pcm_s16le", str(target),
        ]
    )
    if result.returncode != 0:
        raise RuntimeError(f"解码失败 {target.name}: {result.stderr.strip()[:200]}")
    from scipy.io import wavfile

    file_rate, data = wavfile.read(target)
    return {
        "path": display_path(target),
        "sampling_rate": int(file_rate),
        "n_samples": int(data.shape[0]),
        "duration_seconds": round(data.shape[0] / file_rate, 6),
        "sha256": sha256(target),
        "bytes": target.stat().st_size,
    }

def display_path(path: Path, base: Path = ROOT) -> str:
    """尽量给出相对项目根的路径；若路径在项目外（如临时目录）则给出绝对路径。"""

    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)

def tool_versions():
    return {name: run([name, "-version"]).stdout.splitlines()[0] for name in ("ffprobe", "ffmpeg")}
