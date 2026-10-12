# 本程序及代码是在人工智能工具辅助下完成的
# 人工智能工具名称：DeepSeek，型号：DeepSeek V4.1 Flash，开发公司：杭州深度求索人工智能基础技术研究有限公司，版本颁布日期：2026-09-10

"""Map attachment-4 text evidence to its own audio via CTC forced alignment.

Only verified text token-to-transcript matches are assigned candidate seconds.
Audio/vision feature positions are not assumed to have known video timestamps.
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

import csv
import difflib
import hashlib
import json
import pickle
import re
import subprocess
from q1_features.media_io import run as run_media
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC

from q1_features.align_words import (BLANK_ID, MODEL_REVISION, SAMPLING_RATE, SNAPSHOT,
                         WORD_SEPARATOR, build_targets, ctc_forced_align,
                         frame_seconds, load_vocab)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/q3_explanation/q3"
SOURCE = ROOT / "data/attachment4/aligned"
SIMILARITY_HIGH = .90
WORD_CONFIDENCE_MIN = .50


def audio_from_video(path):
    proc = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vn",
                           "-ac", "1", "-ar", str(SAMPLING_RATE),
                           "-f", "f32le", "pipe:1"], capture_output=True, check=True)
    return np.frombuffer(proc.stdout, dtype="<f4").copy()


def align_words(audio, text, model, extractor, vocab, device):
    inputs = extractor(audio, sampling_rate=SAMPLING_RATE, return_tensors="pt")
    with torch.inference_mode():
        logits = model(inputs.input_values.to(device)).logits
        log_probs = torch.log_softmax(logits, dim=-1)[0].cpu().numpy()
    info = build_targets(text, vocab)
    targets = info["target_ids"]
    if not targets:
        return {"status": "no_alignable_text", "words": []}
    try:
        starts, conf, _, collapsed = ctc_forced_align(log_probs, targets)
    except ValueError as exc:
        return {"status": "ctc_alignment_failed", "reason": str(exc), "words": []}
    if collapsed != targets:
        raise AssertionError("CTC alignment self-check failed")
    inverse = {v: k for k, v in vocab.items()}
    greedy = log_probs.argmax(1)
    chars, previous = [], -1
    for token in greedy:
        token = int(token)
        if token != BLANK_ID and token != previous:
            chars.append(inverse.get(token, "?"))
        previous = token
    greedy_text = "".join(chars).replace(WORD_SEPARATOR, " ").strip()
    normalize = lambda s: re.sub(r"[^A-Z' ]", "", s.upper())
    similarity = difflib.SequenceMatcher(None, normalize(greedy_text), normalize(text)).ratio()
    sec = frame_seconds()
    words = []
    for item in info["words"]:
        positions = item["target_positions"]
        if not positions:
            words.append({"word": item["word"], "status": "no_usable_characters"})
            continue
        begin = starts[positions[0]] * sec
        end = (starts[positions[-1]] + 1) * sec
        confidence = float(np.mean([conf[p] for p in positions]))
        status = "candidate" if 0 <= begin < end <= len(audio) / SAMPLING_RATE + .05 else "out_of_audio"
        words.append({"word": item["word"], "start": round(begin, 3),
                      "end": round(end, 3), "confidence": round(confidence, 4),
                      "status": status})
    return {"status": "aligned", "greedy_similarity": round(similarity, 4),
            "greedy_text": greedy_text, "words": words,
            "audio_duration_seconds": round(len(audio) / SAMPLING_RATE, 3)}


def map_tokens(raw_text, token_ids, tokenizer, interval):
    encoded = tokenizer(raw_text, truncation=True, max_length=50, return_offsets_mapping=True)
    valid = token_ids[1].astype(bool)
    if encoded["input_ids"] != token_ids[0, valid].tolist():
        raise ValueError("BERT token IDs do not match original transcript")
    spans = [(match.start(), match.end()) for match in re.finditer(r"\S+", raw_text)]
    begin, end = interval
    selected = []
    for position in range(begin, end + 1):
        a, b = encoded["offset_mapping"][position]
        if a == b:
            continue
        selected.extend(i for i, (x, y) in enumerate(spans) if a < y and b > x)
    if not selected:
        raise ValueError("Evidence tokens have no matching transcript words")
    return min(selected), max(selected)


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(str(SNAPSHOT), local_files_only=True)
    model = Wav2Vec2ForCTC.from_pretrained(str(SNAPSHOT), local_files_only=True).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(str(ROOT / "models/bert-base-uncased"), local_files_only=True)
    vocab = load_vocab()
    full_path = OUT / "attachment4_explanations_full.json"
    csv_path = OUT / "attachment4_predictions_explanations.csv"
    details = json.loads(full_path.read_text(encoding="utf-8"))
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row.setdefault("key_text", "")
        row.setdefault("frame_candidate_seconds", "")
        row.setdefault("frame_candidate_path", "")
        row.setdefault("frame_candidate_status", "")
    lookup = {r["sample_id"]: r for r in rows}
    records = {}
    for count, detail in enumerate(details, 1):
        ident = detail["sample_id"]
        with (SOURCE / f"{ident}.pkl").open("rb") as handle:
            raw = pickle.load(handle)
        audio = audio_from_video(SOURCE / "videos" / f"{ident}.mp4")
        record = align_words(audio, str(raw["raw_text"]), model, extractor, vocab, device)
        records[ident] = record
        if detail["main_modality"] in ("text", "vision") and record["status"] == "aligned":
            modality = detail["main_modality"]
            segment = detail["local"][modality]["high"]
            try:
                first, last = map_tokens(str(raw["raw_text"]), np.asarray(raw["text_bert"]),
                                         tokenizer, (segment["start_position"], segment["end_position"]))
                relevant = record["words"][first:last + 1]
                detail["evidence_transcript_words"] = " ".join(x["word"] for x in relevant)
                detail["evidence_word_indices"] = [first, last]
                if modality == "text":
                    lookup[ident]["key_text"] = detail["evidence_transcript_words"]
                good = (record["greedy_similarity"] >= SIMILARITY_HIGH
                        and all(x["status"] == "candidate" and
                                x["confidence"] >= WORD_CONFIDENCE_MIN for x in relevant))
                if good and modality == "text":
                    start, end = relevant[0]["start"], relevant[-1]["end"]
                    if start < end:
                        detail["evidence_time_seconds"] = [start, end]
                        lookup[ident]["time_start_seconds"] = f"{start:.3f}"
                        lookup[ident]["time_end_seconds"] = f"{end:.3f}"
                        lookup[ident]["time_status"] = "ctc_candidate_requires_video_review"
                elif good and modality == "vision":
                    # Aligned feature positions correspond to token positions, but
                    # the pickle does not contain a measured feature-to-frame map.
                    # Keep this as a cross-modal frame candidate, never a verified
                    # start/end time for the visual feature.
                    midpoint = (relevant[0]["start"] + relevant[-1]["end"]) / 2
                    video = SOURCE / "videos" / f"{ident}.mp4"
                    probe = run_media([
                        "ffprobe", "-v", "error", "-select_streams", "v:0",
                        "-show_entries", "frame=best_effort_timestamp_time",
                        "-of", "json", str(video)]).stdout
                    frame_records = json.loads(probe)["frames"]
                    frames = [float(frame["best_effort_timestamp_time"]) for frame in frame_records]
                    if frames:
                        frame_index = min(range(len(frames)), key=lambda i: abs(frames[i] - midpoint))
                        frame_seconds = frames[frame_index]
                        frame_path = OUT / f"visual_candidate_{ident}.png"
                        subprocess.run(["ffmpeg", "-v", "error", "-i", str(video),
                                        "-vf", f"select=eq(n\\,{frame_index})",
                                        "-frames:v", "1", "-y", str(frame_path)], check=True)
                        lookup[ident]["frame_candidate_seconds"] = f"{frame_seconds:.3f}"
                        lookup[ident]["frame_candidate_path"] = str(frame_path.relative_to(ROOT))
                        lookup[ident]["frame_candidate_status"] = "cross_modal_candidate_requires_review"
                        detail["frame_candidate"] = {"seconds": frame_seconds,
                                                     "path": str(frame_path.relative_to(ROOT)),
                                                     "status": "cross_modal_candidate_requires_review"}
            except (ValueError, IndexError) as exc:
                detail["time_alignment_error"] = str(exc)
        print(f"aligned {count}/20: {ident}, status={record['status']}, similarity={record.get('greedy_similarity')}", flush=True)
    alignment_path = OUT / "attachment4_word_alignment_candidates.json"
    alignment_path.write_text(json.dumps({"model_revision": MODEL_REVISION,
                                          "similarity_threshold": SIMILARITY_HIGH,
                                          "word_confidence_threshold": WORD_CONFIDENCE_MIN,
                                          "records": records}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    full_path.write_text(json.dumps(details, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    digest = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    manifest_path = OUT / "attachment4_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["csv_sha256"] = digest
    manifest["alignment_candidates"] = str(alignment_path.relative_to(ROOT))
    manifest["n_candidate_text_times"] = sum(r["time_status"].startswith("ctc_candidate") for r in rows)
    manifest["n_candidate_video_frames"] = sum(bool(r["frame_candidate_path"]) for r in rows)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"candidate_text_times": sum(r["time_status"].startswith("ctc_candidate") for r in rows),
                      "n_rows": len(rows), "output": str(csv_path.relative_to(ROOT))}, ensure_ascii=False))


if __name__ == "__main__":
    main()
