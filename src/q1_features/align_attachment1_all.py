# 本程序及代码是在人工智能工具辅助下完成的
# 人工智能工具名称：DeepSeek，型号：DeepSeek V4.1 Flash，开发公司：杭州深度求索人工智能基础技术研究有限公司，版本颁布日期：2026-09-10

"""Generate quality-gated word-time candidates for all 100 attachment 1 clips.

This does not change the accepted feature NPZ files: candidate times need review.
"""

from __future__ import annotations
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
for _p in (str(HERE.parent), str(HERE)):          # src/ 与本文件所在目录
    if _p not in sys.path:
        sys.path.insert(0, _p)


import csv
import json
from pathlib import Path

import numpy as np
import torch
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC

from align_words import (AUDIO_DIR, INVENTORY_CSV, MODEL_ID, MODEL_REVISION,
                         ROOT, SNAPSHOT, decode_audio, frame_seconds, load_vocab, run_alignment)
from feature_extractors import safe_name

OUTPUT = ROOT / "outputs/q1_features/attachment1_alignment/all_candidates"


def main() -> None:
    if not SNAPSHOT.exists():
        raise SystemExit(f"Missing {MODEL_ID}@{MODEL_REVISION}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    vocab = load_vocab()
    step = frame_seconds()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(str(SNAPSHOT))
    model = Wav2Vec2ForCTC.from_pretrained(str(SNAPSHOT)).to(device).eval()
    with INVENTORY_CSV.open(encoding="utf-8") as handle:
        samples = list(csv.DictReader(handle))
    rows = []
    for index, sample in enumerate(samples, 1):
        key = safe_name(sample["stable_id"])
        path = OUTPUT / f"{key}.json"
        try:
            # Cached candidates do not imply that disposable decoded audio exists.
            wav_path = AUDIO_DIR / f"{key}.wav"
            if not wav_path.exists():
                decode_audio(ROOT / sample["relative_path"], wav_path)
            if path.exists():
                record = json.loads(path.read_text(encoding="utf-8"))
            else:
                timebase = json.loads((ROOT / f"outputs/q1_features/attachment1_timebase/per_sample/{key}.json").read_text(encoding="utf-8"))
                frame_times = timebase["video_frames"]["display_times"]
                record = run_alignment(key, sample, vocab, step, frame_times,
                                       (extractor, model, device))
                path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            quality = record.get("quality", {})
            row = {"sample_id": sample["stable_id"], "status": record.get("status"),
                   "alignment_status": record.get("alignment_status"),
                   "n_words": record.get("n_words", 0),
                   "n_path_found": quality.get("n_ctc_path_found", 0),
                   "n_auto_usable_candidates": quality.get("n_words_usable_for_feature_alignment", 0),
                   "char_similarity": record.get("diagnostic_asr", {}).get("char_similarity_to_official"),
                   "confidence_mean": quality.get("confidence_mean"),
                   "n_words_in_silence": quality.get("n_words_entirely_in_silence", 0),
                   "error": record.get("error")}
        except Exception as error:
            row = {"sample_id": sample["stable_id"], "status": "failed_exception",
                   "error": f"{type(error).__name__}: {error}"}
            (OUTPUT / f"{key}_error.json").write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        rows.append(row)
        print(f"[{index}/{len(samples)}] {row['sample_id']} {row['status']} "
              f"{row.get('alignment_status')} {row.get('n_auto_usable_candidates', 0)}/{row.get('n_words', 0)}",
              flush=True)
    with (OUTPUT / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {"n_samples": len(rows), "model": f"{MODEL_ID}@{MODEL_REVISION}",
               "status_counts": {x: sum(r.get("alignment_status") == x for r in rows)
                                 for x in sorted({r.get("alignment_status") for r in rows}, key=str)},
               "n_words": int(sum(r.get("n_words", 0) for r in rows)),
               "n_path_found": int(sum(r.get("n_path_found", 0) for r in rows)),
               "n_auto_usable_candidates": int(sum(r.get("n_auto_usable_candidates", 0) for r in rows)),
               "note": "Times are automatic candidates; they are not promoted to formal features without manual review."}
    (OUTPUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
