# 本程序及代码是在人工智能工具辅助下完成的
# 人工智能工具名称：DeepSeek，型号：DeepSeek V4.1 Flash，开发公司：杭州深度求索人工智能基础技术研究有限公司，版本颁布日期：2026-09-10

"""冻结CTC模型的词时间候选、路径求解与独立穷举验证。"""

from __future__ import annotations

import json

import re

import subprocess

from pathlib import Path

import numpy as np

from scipy.io import wavfile

ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = ROOT / "outputs" / "q1_features" / "attachment1_alignment"

AUDIO_DIR = OUTPUT_DIR / "audio"

INVENTORY_CSV = ROOT / "outputs" / "q1_features" / "attachment1_inventory.csv"

MODEL_ID = "facebook/wav2vec2-base-960h"

MODEL_REVISION = "22aad52d435eb6dbaf354bdad9b0da84ce7d6156"

SNAPSHOT = (
    ROOT
    / "models/wav2vec2-base-960h"
)

FFMPEG = "ffmpeg"

SAMPLING_RATE = 16000

BLANK_ID = 0

WORD_SEPARATOR = "|"

ALLOWED_CHARS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ'")

LOW_CONFIDENCE_THRESHOLD = 0.5

FAILED_CONFIDENCE_THRESHOLD = 0.10

SAMPLE_STATUS_CANDIDATE = "candidate_for_manual_spot_check"

SAMPLE_STATUS_NEEDS_REVIEW = "needs_review"

SAMPLE_STATUS_NOT_USABLE = "not_usable_for_formal_alignment"

WORD_STATUS_USABLE = "usable_candidate"

WORD_STATUS_NEEDS_REVIEW = "needs_review"

WORD_STATUS_NOT_USABLE = "not_usable"

WORD_STATUS_FAILED = "failed"

SIMILARITY_HIGH = 0.90

SIMILARITY_LOW = 0.20

SILENCE_NOISE = "-35dB"

SILENCE_MIN_DURATION = 0.15

def load_vocab() -> dict:
    vocab = json.loads((SNAPSHOT / "vocab.json").read_text(encoding="utf-8"))
    return vocab

def frame_seconds() -> float:
    config = json.loads((SNAPSHOT / "config.json").read_text(encoding="utf-8"))
    stride = 1
    for value in config["conv_stride"]:
        stride *= int(value)
    return stride / SAMPLING_RATE

def pronounce(word: str, vocab: dict) -> tuple[list[int], list[str], bool]:
    """词 -> (字符 id 列表, 丢弃字符列表, 是否全部字符都保留)。"""

    kept_ids: list[int] = []
    dropped: list[str] = []
    for char in word:
        if char in ALLOWED_CHARS and char in vocab:
            kept_ids.append(vocab[char])
        else:
            dropped.append(char)
    return kept_ids, dropped, len(kept_ids) > 0

def build_targets(text: str, vocab: dict) -> dict:
    """把官方转写切成词并构造含词分隔符的目标序列。

    词内标点等不可能映射的字符丢弃并逐词记录；只要词内还有至少一个可映射字符，
    该词就参与对齐（不因标点而整词失败）。
    """

    raw_words = [w for w in text.split() if w]
    word_items = []
    target_ids: list[int] = []
    for index, raw in enumerate(raw_words):
        upper = raw.upper()
        ids, dropped, _ = pronounce(upper, vocab)
        if index > 0:
            target_ids.append(vocab[WORD_SEPARATOR])
        start_pos = len(target_ids)
        target_ids.extend(ids)
        word_items.append(
            {
                "word_index": index,
                "word": raw,
                "word_upper": upper,
                "n_chars_kept": len(ids),
                "chars_dropped": dropped,
                "usable_for_alignment": len(ids) > 0,
                "fully_representable": len(ids) == len(upper) and len(upper) > 0,
                "target_positions": list(range(start_pos, start_pos + len(ids))),
            }
        )
    return {"words": word_items, "target_ids": target_ids}

def decode_audio(video: Path, wav_path: Path) -> None:
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            FFMPEG, "-v", "error", "-y", "-i", str(video),
            "-vn", "-ac", "1", "-ar", str(SAMPLING_RATE), "-c:a", "pcm_s16le", str(wav_path),
        ],
        check=True,
    )

def detect_silence(wav_path: Path) -> list[dict]:
    result = subprocess.run(
        [
            FFMPEG, "-v", "info", "-i", str(wav_path),
            "-af", f"silencedetect=noise={SILENCE_NOISE}:d={SILENCE_MIN_DURATION}",
            "-f", "null", "-",
        ],
        capture_output=True,
        text=True,
    )
    segments = []
    pending_start = None
    for line in result.stderr.splitlines():
        m = re.search(r"silence_start:\s*(-?[\d.]+)", line)
        if m:
            pending_start = float(m.group(1))
        m = re.search(r"silence_end:\s*(-?[\d.]+)\s*\|\s*silence_duration:\s*([\d.]+)", line)
        if m:
            end = float(m.group(1))
            segments.append(
                {
                    "start": max(0.0, pending_start if pending_start is not None else end - float(m.group(2))),
                    "end": end,
                    "duration": float(m.group(2)),
                }
            )
            pending_start = None
    if pending_start is not None:
        segments.append({"start": max(0.0, pending_start), "end": None, "duration": None, "open_ended": True})
    for seg in segments:
        seg["start"] = round(seg["start"], 6)
        if seg["end"] is not None:
            seg["end"] = round(seg["end"], 6)
    return {
        "params": {"noise": SILENCE_NOISE, "min_duration_s": SILENCE_MIN_DURATION, "filter": "silencedetect"},
        "segments": segments,
    }

def min_required_frames(targets: list[int]) -> int:
    """CTC 强制对齐所需最少帧数：每个目标字符至少 1 帧，且相邻相同字符之间必须有 blank。"""

    repeats = sum(1 for i in range(1, len(targets)) if targets[i] == targets[i - 1])
    return len(targets) + repeats

def ctc_forced_align(log_probs: np.ndarray, targets: list[int]) -> tuple[list[int], list[float], float, list[int]]:
    """CTC Viterbi 强制对齐。

    返回 (每个目标字符的起始帧, 每个目标字符的置信度, 最优路径总对数概率, 回解得到的词元序列)。
    回解序列用于自检：它必须等于输入目标序列。
    log_probs: (T, V)
    """

    T, _ = log_probs.shape
    N = len(targets)
    if N == 0:
        raise ValueError("目标序列为空，无法对齐")
    required = min_required_frames(targets)
    if T < required:
        # 帧数不足时最优路径会退化为全 blank，必须显式失败，不能返回无意义的时间
        raise ValueError(
            f"声学帧数 {T} 少于该目标序列所需最少帧数 {required}（含相邻重复字符间的 blank），无法强制对齐"
        )
    extended = [BLANK_ID]
    for token in targets:
        extended.append(token)
        extended.append(BLANK_ID)
    E = len(extended)

    NEG = -1e30
    alpha = np.full((T, E), NEG, dtype=np.float64)
    back = np.zeros((T, E), dtype=np.int32)
    alpha[0, 0] = log_probs[0, BLANK_ID]
    if E > 1:
        alpha[0, 1] = log_probs[0, extended[1]]

    for t in range(1, T):
        emission = log_probs[t]
        for s in range(E):
            best = alpha[t - 1, s]
            src = s
            if s - 1 >= 0 and alpha[t - 1, s - 1] > best:
                best = alpha[t - 1, s - 1]
                src = s - 1
            # 跳过中间 blank（仅在 s 与 s-2 都是非 blank 且不同字符时允许）
            if (
                s - 2 >= 0
                and extended[s] != BLANK_ID
                and extended[s] != extended[s - 2]
                and alpha[t - 1, s - 2] > best
            ):
                best = alpha[t - 1, s - 2]
                src = s - 2
            alpha[t, s] = best + emission[extended[s]] if best > NEG / 2 else NEG
            back[t, s] = src

    end = E - 1
    if E >= 2 and alpha[T - 1, E - 2] > alpha[T - 1, end]:
        end = E - 2
    total = float(alpha[T - 1, end])
    if total <= -1e29:
        raise ValueError("目标序列在该音频上不可达，强制对齐失败")

    path = np.zeros(T, dtype=np.int32)
    state = end
    for t in range(T - 1, -1, -1):
        path[t] = state
        state = int(back[t, state])

    starts = [-1] * N
    ends = [-1] * N
    for t in range(T):
        s = int(path[t])
        if s % 2 == 1:
            token_index = (s - 1) // 2
            if starts[token_index] < 0:
                starts[token_index] = t
            ends[token_index] = t

    confidences = []
    for i, token in enumerate(targets):
        if starts[i] < 0:
            confidences.append(0.0)
            continue
        frames = log_probs[starts[i] : ends[i] + 1, token]
        confidences.append(float(np.exp(frames.mean())))

    # 回解：路径 -> 折叠重复与 blank -> 词元序列（用于自检）
    collapsed: list[int] = []
    previous_token = None
    for t in range(T):
        s = int(path[t])
        if s % 2 == 1:
            token = extended[s]
            if token != previous_token:
                collapsed.append(token)
            previous_token = token
        else:
            previous_token = None
    return starts, confidences, total, collapsed

def _extended_sequence(targets: list[int]) -> list[int]:
    extended = [BLANK_ID]
    for token in targets:
        extended.append(token)
        extended.append(BLANK_ID)
    return extended

def _successors(state: int, extended: list[int]) -> list[int]:
    """合法转移：停留、前进 1、或（当前与 s+2 均为非 blank 且字符不同时）前进 2。"""

    out = [state]
    if state + 1 < len(extended):
        out.append(state + 1)
    if (
        state + 2 < len(extended)
        and extended[state + 2] != BLANK_ID
        and extended[state + 2] != extended[state]
    ):
        out.append(state + 2)
    return out

def enumerate_legal_paths(
    log_probs: np.ndarray,
    targets: list[int],
    cap: int = 500_000,
) -> dict:
    """**真穷举**全部合法 CTC 状态路径（不使用动态规划）的最优对数概率。

    合法路径：首状态 ∈ {0, 1}；每步按 :func:`_successors` 转移；
    末状态必须是末尾 blank ``E-1`` 或最后一个目标字符 ``E-2``。

    返回 ``{best_logprob, n_paths, capped, best_path}``；若没有任何合法路径，
    ``n_paths == 0`` 且 ``best_logprob == -inf``。路径数超过 ``cap`` 时 ``capped=True``。
    """

    T = log_probs.shape[0]
    if T == 0 or not targets:
        return {"best_logprob": float("-inf"), "n_paths": 0, "capped": False, "best_path": None}

    extended = _extended_sequence(targets)
    E = len(extended)
    best = float("-inf")
    best_path: list[int] | None = None
    state = {"n": 0, "capped": False}

    def walk(t: int, node: int, path: list[int], acc: float) -> None:
        nonlocal best, best_path
        if state["capped"]:
            return
        if t == T:
            if node in (E - 1, E - 2):
                state["n"] += 1
                if state["n"] > cap:
                    state["capped"] = True
                    return
                if acc > best:
                    best = acc
                    best_path = list(path)
            return
        for nxt in _successors(node, extended):
            path.append(nxt)
            walk(t + 1, nxt, path, acc + float(log_probs[t, extended[nxt]]))
            path.pop()
            if state["capped"]:
                return

    starts = (0, 1) if E > 1 else (0,)
    for start in starts:
        walk(1, start, [start], float(log_probs[0, extended[start]]))
        if state["capped"]:
            break

    return {
        "best_logprob": best,
        "n_paths": state["n"],
        "capped": state["capped"],
        "best_path": best_path,
    }

def run_alignment(label: str, sample: dict, vocab: dict, frame_sec: float,
                  video_frames: list[float], model_bundle: tuple | None = None) -> dict:
    video = ROOT / sample["relative_path"]
    wav_path = AUDIO_DIR / f"{label}.wav"
    if not wav_path.exists():
        decode_audio(video, wav_path)

    rate, wave = wavfile.read(wav_path)
    if wave.ndim > 1:
        wave = wave.mean(axis=1)
    audio = wave.astype(np.float32) / 32768.0
    audio_duration = len(audio) / rate

    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC
    import torch

    if model_bundle is None:
        extractor = Wav2Vec2FeatureExtractor.from_pretrained(str(SNAPSHOT))
        model = Wav2Vec2ForCTC.from_pretrained(str(SNAPSHOT)).eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
    else:
        extractor, model, device = model_bundle

    inputs = extractor(audio, sampling_rate=SAMPLING_RATE, return_tensors="pt")
    with torch.no_grad():
        logits = model(inputs.input_values.to(device)).logits
        log_probs = torch.log_softmax(logits, dim=-1)[0].cpu().numpy()

    n_frames = log_probs.shape[0]
    targets_info = build_targets(sample["text"], vocab)
    targets = targets_info["target_ids"]

    record: dict = {
        "label": label,
        "stable_id": sample["stable_id"],
        "relative_path": sample["relative_path"],
        "official_text": sample["text"],
        "n_words": len(targets_info["words"]),
        "audio": {
            "wav_path": str(wav_path.relative_to(ROOT)),
            "sampling_rate": rate,
            "n_samples": int(len(audio)),
            "duration_seconds": round(audio_duration, 6),
        },
        "index_to_seconds": {
            "audio_sample_to_seconds": "sample_index / 16000",
            "acoustic_frame_to_seconds": f"frame_index * {frame_sec}",
            "frame_stride_samples": int(round(frame_sec * SAMPLING_RATE)),
            "acoustic_frames": int(n_frames),
            "emission_span_seconds": round(n_frames * frame_sec, 6),
        },
    }

    if len(targets) == 0:
        record["status"] = "failed_no_target_characters"
        record["words"] = []
        return record

    try:
        starts, confidences, total_logprob, collapsed = ctc_forced_align(log_probs, targets)
    except ValueError as error:
        record["status"] = "failed_alignment"
        record["error"] = str(error)
        record["words"] = []
        return record

    record["self_check"] = {
        "recovered_sequence_equals_target": collapsed == targets,
        "n_target_tokens": len(targets),
        "min_required_frames": min_required_frames(targets),
    }

    # 诊断量：模型自身的自由解码文本与官方文本的相似度。
    # 仅用于解释低置信度（例如官方转写与该音频不匹配），绝不用作对齐目标。
    greedy_ids = log_probs.argmax(axis=1)
    greedy_chars: list[str] = []
    previous = -1
    inverse_vocab = {v: k for k, v in vocab.items()}
    for token in greedy_ids:
        token = int(token)
        if token != BLANK_ID and token != previous:
            greedy_chars.append(inverse_vocab.get(token, "?"))
        previous = token
    greedy_text = "".join(greedy_chars).replace(WORD_SEPARATOR, " ").strip()
    import difflib

    official_upper = " ".join(sample["text"].upper().split())
    similarity = difflib.SequenceMatcher(
        None, re.sub(r"[^A-Z' ]", "", greedy_text), re.sub(r"[^A-Z' ]", "", official_upper)
    ).ratio()
    record["diagnostic_asr"] = {
        "note": "仅用于诊断文本与音频是否匹配，不作为对齐目标，也不替换官方转写",
        "greedy_text": greedy_text,
        "char_similarity_to_official": round(float(similarity), 6),
    }

    words = []
    for item in targets_info["words"]:
        positions = item["target_positions"]
        if item["usable_for_alignment"] and positions:
            begin_frame = starts[positions[0]]
            end_frame = starts[positions[-1]]
            word_confs = [confidences[p] for p in positions]
            if begin_frame < 0 or end_frame < 0:
                words.append({**item, "status": "failed_no_frames"})
                continue
            start_sec = begin_frame * frame_sec
            end_sec = (end_frame + 1) * frame_sec
            words.append(
                {
                    "word_index": item["word_index"],
                    "word": item["word"],
                    "start": round(start_sec, 6),
                    "end": round(end_sec, 6),
                    "duration": round(end_sec - start_sec, 6),
                    "start_frame": int(begin_frame),
                    "end_frame": int(end_frame),
                    "confidence_mean": round(float(np.mean(word_confs)), 6),
                    "confidence_min": round(float(np.min(word_confs)), 6),
                    "n_chars_kept": item["n_chars_kept"],
                    "chars_dropped": item["chars_dropped"],
                    "status": "ctc_path_found",
                }
            )
        else:
            words.append(
                {
                    "word_index": item["word_index"],
                    "word": item["word"],
                    "status": "failed_no_usable_characters",
                    "chars_dropped": item["chars_dropped"],
                }
            )

    aligned = [w for w in words if w.get("status") == "ctc_path_found"]
    starts_sec = [w["start"] for w in aligned]
    monotonic = all(starts_sec[i] > starts_sec[i - 1] for i in range(1, len(starts_sec)))
    audio_span_limit = max(audio_duration, n_frames * frame_sec)
    video_span = (min(video_frames), max(video_frames)) if video_frames else (None, None)
    out_of_audio = [w["word"] for w in aligned if w["start"] < 0 or w["end"] > audio_span_limit + 1e-6]
    out_of_video = []
    if video_frames:
        out_of_video = [
            w["word"] for w in aligned if w["end"] > video_span[1] + 0.05 or w["start"] < video_span[0] - 0.05
        ]

    silence = detect_silence(wav_path)
    silence_segments = [s for s in silence["segments"] if s.get("end") is not None]

    def in_silence(word: dict) -> bool:
        for seg in silence_segments:
            if word["start"] >= seg["start"] - 1e-9 and word["end"] <= seg["end"] + 1e-9:
                return True
        return False

    for word in aligned:
        word["entirely_in_silence"] = in_silence(word)
        word["low_confidence"] = word["confidence_mean"] < LOW_CONFIDENCE_THRESHOLD
        word["failed_confidence"] = word["confidence_mean"] < FAILED_CONFIDENCE_THRESHOLD

    confidences_all = [w["confidence_mean"] for w in aligned]
    n_aligned = len(aligned)
    low_fraction = (
        sum(1 for w in aligned if w["low_confidence"]) / n_aligned if n_aligned else 1.0
    )
    failed_fraction = (
        sum(1 for w in aligned if w["failed_confidence"]) / n_aligned if n_aligned else 1.0
    )
    similarity = record["diagnostic_asr"]["char_similarity_to_official"]

    # --- 样本级质量状态：严格区分“CTC 路径求出”与“时间可用于特征对齐” ---
    reasons: list[str] = []
    if similarity < SIMILARITY_LOW:
        # 主判据：英文声学模型与该段音频的官方文本在字符层面几乎不相容，
        # 此时“求出的 CTC 路径”没有意义，词时间不得用于正式对齐。
        reasons.append(
            f"诊断字符相似度 {similarity:.3f} < {SIMILARITY_LOW}：英文声学模型与该段音频的官方文本不相容"
        )
        sample_status = SAMPLE_STATUS_NOT_USABLE
    elif similarity < SIMILARITY_HIGH or low_fraction > 0:
        if similarity < SIMILARITY_HIGH:
            reasons.append(f"诊断字符相似度 {similarity:.3f} < {SIMILARITY_HIGH}")
        if low_fraction > 0:
            reasons.append(
                f"存在低置信词：占比 {low_fraction:.3f}（失败置信占比 {failed_fraction:.3f}）"
            )
        sample_status = SAMPLE_STATUS_NEEDS_REVIEW
    else:
        reasons.append(
            f"诊断字符相似度 {similarity:.3f} ≥ {SIMILARITY_HIGH} 且无低置信词；"
            "仍需人工抽查后才可用于正式对齐"
        )
        sample_status = SAMPLE_STATUS_CANDIDATE

    # --- 词级质量状态 ---
    for word in words:
        if word.get("status") != "ctc_path_found":
            word["time_status"] = WORD_STATUS_FAILED
            word["usable_for_feature_alignment"] = False
            word["exclusion_reason"] = "未求出 CTC 路径（字符不可用或帧数不足）"
            continue
        confidence = word["confidence_mean"]
        if confidence < FAILED_CONFIDENCE_THRESHOLD:
            word["time_status"] = WORD_STATUS_NOT_USABLE
        elif confidence < LOW_CONFIDENCE_THRESHOLD:
            word["time_status"] = WORD_STATUS_NEEDS_REVIEW
        else:
            word["time_status"] = WORD_STATUS_USABLE
        if sample_status != SAMPLE_STATUS_CANDIDATE:
            word["usable_for_feature_alignment"] = False
            word["exclusion_reason"] = f"样本级状态为 {sample_status}（{reasons[0]}）"
        elif word["time_status"] != WORD_STATUS_USABLE:
            word["usable_for_feature_alignment"] = False
            word["exclusion_reason"] = f"词级状态为 {word['time_status']}（置信度 {confidence:.3f}）"
        else:
            word["usable_for_feature_alignment"] = True
            word["exclusion_reason"] = None

    usable_words = [w for w in words if w.get("usable_for_feature_alignment")]
    record.update(
        {
            "status": "ctc_path_found",
            "alignment_status": sample_status,
            "alignment_status_reason": reasons,
            "alignment_status_criteria": {
                "primary_rule": (
                    f"诊断字符相似度 < {SIMILARITY_LOW} -> not_usable_for_formal_alignment（英文模型与官方文本不相容）；"
                    f"相似度 < {SIMILARITY_HIGH} 或存在低置信词 -> needs_review；"
                    f"相似度 >= {SIMILARITY_HIGH} 且无低置信词 -> candidate_for_manual_spot_check"
                ),
                "similarity_high": SIMILARITY_HIGH,
                "similarity_low": SIMILARITY_LOW,
                "low_confidence_threshold": LOW_CONFIDENCE_THRESHOLD,
                "failed_confidence_threshold": FAILED_CONFIDENCE_THRESHOLD,
                "low_confidence_fraction": round(low_fraction, 6),
                "failed_confidence_fraction": round(failed_fraction, 6),
                "similarity_role": (
                    "仅用于判断“英文声学模型与官方文本是否相容”，"
                    "不作为官方标签正确性、视频配对正确性或语音语种判断的证据"
                ),
                "word_usable_rule": "只有样本级状态为 candidate_for_manual_spot_check 且词级状态为 usable_candidate 的词，才允许进入后续特征对齐",
                "note": "固定筛查阈值仅依据附件一设定，不使用附件二、三、四选择阈值",
            },
            "words_usable_for_feature_alignment": [w["word_index"] for w in usable_words],
            "words": words,
            "silence": silence,
            "checks": {
                "ctc_path_found_for_all_words": len(aligned) == len(words),
                "word_start_times_monotonic_increasing": bool(monotonic),
                "words_out_of_audio_range": out_of_audio,
                "words_out_of_video_display_range": out_of_video,
                "audio_span_seconds": round(audio_duration, 6),
                "video_display_span_seconds": [round(video_span[0], 6), round(video_span[1], 6)]
                if video_frames
                else None,
                "total_path_logprob": round(total_logprob, 6),
                "mean_word_logprob": round(total_logprob / max(1, len(targets)), 6),
                "video_frame_offset_is_not_accuracy_evidence": (
                    "词时间与邻近视频帧时间的差只说明“如何从候选时间找到附近视频帧”，"
                    "不构成词边界准确性的证据"
                ),
            },
            "quality": {
                "n_words": len(words),
                "n_ctc_path_found": len(aligned),
                "ctc_path_found_fraction": round(len(aligned) / len(words), 6) if words else 0.0,
                "word_coverage": round(len(aligned) / len(words), 6) if words else 0.0,
                "quality_status": sample_status,
                "quality_status_reason": reasons,
                "n_words_usable_for_feature_alignment": len(usable_words),
                "low_confidence_fraction": round(low_fraction, 6),
                "failed_confidence_fraction": round(failed_fraction, 6),
                "n_failed_no_usable_chars": sum(1 for w in words if w.get("status") == "failed_no_usable_characters"),
                "n_failed_no_frames": sum(1 for w in words if w.get("status") == "failed_no_frames"),
                "confidence_mean": round(float(np.mean(confidences_all)), 6) if confidences_all else None,
                "confidence_min": round(float(np.min(confidences_all)), 6) if confidences_all else None,
                "confidence_median": round(float(np.median(confidences_all)), 6) if confidences_all else None,
                "n_low_confidence": sum(1 for w in aligned if w["low_confidence"]),
                "n_failed_confidence": sum(1 for w in aligned if w["failed_confidence"]),
                "n_words_entirely_in_silence": sum(1 for w in aligned if w["entirely_in_silence"]),
                "diagnostic_char_similarity_to_official": similarity,
                "mean_word_logprob": round(total_logprob / max(1, len(targets)), 6),
                "thresholds": {
                    "low_confidence": LOW_CONFIDENCE_THRESHOLD,
                    "failed_confidence": FAILED_CONFIDENCE_THRESHOLD,
                    "note": "报告用固定阈值，不参与任何调参，未在附件 3、4 上选择",
                },
            },
        }
    )
    return record
