# 本程序及代码是在人工智能工具辅助下完成的
# 人工智能工具名称：DeepSeek，型号：DeepSeek V4.1 Flash，开发公司：杭州深度求索人工智能基础技术研究有限公司，版本颁布日期：2026-09-10

"""冻结BERT文本编码与逐帧面部特征提取。"""

from __future__ import annotations

import re

import subprocess

import tempfile

from pathlib import Path

import numpy as np

from audio_features import FILL_VALUE

BLENDSHAPE_NAMES_52 = sorted([
    "_neutral",
    "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight",
    "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
    "eyeBlinkLeft", "eyeBlinkRight",
    "eyeLookDownLeft", "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight",
    "eyeLookOutLeft", "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight",
    "eyeSquintLeft", "eyeSquintRight", "eyeWideLeft", "eyeWideRight",
    "jawForward", "jawLeft", "jawOpen", "jawRight",
    "mouthClose", "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft", "mouthFrownRight",
    "mouthFunnel", "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthPressLeft", "mouthPressRight", "mouthPucker", "mouthRight",
    "mouthRollLower", "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper",
    "mouthSmileLeft", "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight",
    "mouthUpperUpLeft", "mouthUpperUpRight", "noseSneerLeft", "noseSneerRight",
])

def safe_name(stable_id: str) -> str:
    """与时间基准核验产物一致的命名：video_id__clip_id（保留前导减号，避免碰撞）。"""

    return stable_id.replace("$_$", "__")

ROOT = Path(__file__).resolve().parents[2]

BERT_DIR = ROOT / "models" / "bert-base-uncased"

FACE_MODEL = ROOT / "models" / "mediapipe" / "face_landmarker.task"

FFMPEG = "ffmpeg"

VISION_WIDTH = 640

N_BLENDSHAPES = 52

MAX_TEXT_TOKENS = 512

NUM_FACES = 2

def extract_frames(video: Path, indices: np.ndarray, width: int = VISION_WIDTH,
                   expected_height: int | None = None) -> tuple[list[np.ndarray], list[float], list[str]]:
    """用 ffmpeg 只输出指定显示序帧号的 RGB 帧；同时用 showinfo 读回真实 pts_time。"""

    if indices.size == 0:
        return [], [], []
    # Collapse consecutive frame indices so long clips do not exceed the
    # filter expression parser's recursion limit.
    indices = np.asarray(indices, dtype=np.int64)
    if np.any(indices < 0) or np.any(np.diff(indices) <= 0):
        raise ValueError("Frame indices must be nonnegative and strictly increasing")
    runs = np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
    terms = [f"eq(n\\,{int(run[0])})" if len(run) == 1 else
             f"between(n\\,{int(run[0])}\\,{int(run[-1])})" for run in runs]
    while len(terms) > 1:
        terms = [f"({terms[i]}+{terms[i+1]})" if i+1 < len(terms) else terms[i]
                 for i in range(0, len(terms), 2)]
    select_expr = terms[0]
    vf = f"select='{select_expr}',scale={width}:-2,showinfo"
    with tempfile.TemporaryFile() as errfile:
        process = subprocess.Popen(
            [FFMPEG, "-v", "info", "-i", str(video), "-vf", vf, "-fps_mode", "passthrough",
             "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
            stdout=subprocess.PIPE, stderr=errfile,
        )
        raw = process.stdout.read()
        process.wait()
        errfile.seek(0)
        stderr_text = errfile.read().decode("utf-8", "replace")
    if process.returncode != 0:
        raise RuntimeError(f"抽帧失败: {stderr_text.strip().splitlines()[-1][:200] if stderr_text else ''}")

    height = expected_height
    if height is None:
        if len(raw) % (indices.size * width * 3) != 0:
            raise RuntimeError(
                f"原始帧字节数 {len(raw)} 与 {indices.size} 帧 × {width} 宽不匹配，无法确定高度"
            )
        height = len(raw) // (indices.size * width * 3)
    frame_bytes = width * height * 3
    if len(raw) != indices.size * frame_bytes:
        raise RuntimeError(
            f"帧字节数不匹配：期望 {indices.size * frame_bytes}，实际 {len(raw)}"
            f"（宽 {width} 高 {height}）"
        )
    frames = [np.frombuffer(raw, dtype=np.uint8, count=frame_bytes, offset=k * frame_bytes)
              .reshape(height, width, 3) for k in range(indices.size)]
    pts = [float(m) for m in re.findall(r"pts_time:([0-9.]+)", stderr_text)]
    if len(pts) != indices.size:
        pts = []
    return frames, pts, stderr_text.splitlines()

def blendshape_names() -> list[str]:
    """没有任何人脸时也要有固定的 52 名称序，避免整条样本失败。"""

    if _blendshape_name_cache:
        return list(_blendshape_name_cache)
    return list(BLENDSHAPE_NAMES_52)

_blendshape_name_cache: list[str] = []

def run_face_landmarker(video: Path, indices: np.ndarray, display_times: np.ndarray,
                        width: int = VISION_WIDTH, height: int = 360) -> dict:
    """对抽出的帧跑 Face Landmarker（VIDEO 模式），返回 blendshape 与检测状态。"""

    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    frames, pts, _ = extract_frames(video, indices, width=width, expected_height=height)
    actual_times = np.asarray(pts, dtype=np.float64) if pts else display_times[indices]

    options = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(FACE_MODEL)),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=NUM_FACES,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=False,
    )
    landmarker = vision.FaceLandmarker.create_from_options(options)

    features = np.full((frames.__len__(), N_BLENDSHAPES), FILL_VALUE, dtype=np.float64)
    face_observed = np.zeros(frames.__len__(), dtype=bool)
    n_faces = np.zeros(frames.__len__(), dtype=np.int32)
    names: list[str] = []
    try:
        for k, frame in enumerate(frames):
            # extract_frames 请求 ffmpeg 输出 rgb24；这里已经是 RGB，不能再交换通道。
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame))
            timestamp_ms = int(round(float(actual_times[k]) * 1000.0))
            if k > 0:
                previous = int(round(float(actual_times[k - 1]) * 1000.0))
                if timestamp_ms <= previous:
                    timestamp_ms = previous + 1
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            blendshapes = result.face_blendshapes or []
            n_faces[k] = len(blendshapes)
            if blendshapes:
                categories = sorted(blendshapes[0], key=lambda c: c.category_name)
                if not names:
                    names = [c.category_name for c in categories]
                    if len(_blendshape_name_cache) != len(names):
                        # 幂等赋值：不能 extend，否则跨样本累积会把缓存撑大
                        _blendshape_name_cache[:] = names
                features[k] = [c.score for c in categories]
                face_observed[k] = True
    finally:
        landmarker.close()

    if not names:
        names = blendshape_names()
    if len(names) != N_BLENDSHAPES:
        names = list(BLENDSHAPE_NAMES_52)
    return {
        "features": features,
        "face_observed": face_observed,
        "n_faces": n_faces,
        "blendshape_names": names,
        "display_times": actual_times,
        "source_frame_indices": indices,
        "n_frames_extracted": len(frames),
    }

def encode_text(text: str, tokenizer, model, device) -> dict:
    import torch

    # Pretokenize by official whitespace words so word_ids refer to the same
    # word indices as the CTC alignment and human review tables. Token IDs are
    # unchanged for all 100 supplied transcripts (verified before migration).
    words = text.split()
    encoding = tokenizer(words, is_split_into_words=True, add_special_tokens=True, truncation=True,
                         max_length=MAX_TEXT_TOKENS, return_tensors="pt")
    input_ids = encoding["input_ids"][0]
    attention = encoding["attention_mask"][0]
    token_type = encoding.get("token_type_ids", torch.zeros_like(input_ids))[0]
    n_before = len(tokenizer(words, is_split_into_words=True, add_special_tokens=True,
                             truncation=False)["input_ids"])
    with torch.no_grad():
        outputs = model(input_ids=input_ids.unsqueeze(0).to(device),
                        attention_mask=attention.unsqueeze(0).to(device))
    hidden = outputs.last_hidden_state[0].cpu().numpy().astype(np.float32)
    # 词 → 子词映射（用快速分词器的 word_ids）
    word_ids = encoding.word_ids(0) if hasattr(encoding, "word_ids") else None
    if word_ids is None:
        word_ids = tokenizer(words, is_split_into_words=True, add_special_tokens=True,
                             truncation=True, max_length=MAX_TEXT_TOKENS).word_ids()
    mapping = [-1 if w is None else int(w) for w in word_ids]
    return {
        "input_ids": input_ids.numpy().astype(np.int32),
        "attention_mask": attention.numpy().astype(bool),
        "token_type_ids": token_type.numpy().astype(np.int32),
        "last_hidden_state": hidden,
        "cls_vector": hidden[0].copy(),
        "word_ids": np.asarray(mapping, dtype=np.int32),
        "n_tokens": int(input_ids.shape[0]),
        "n_tokens_before_truncation": int(n_before),
        "truncated": bool(n_before > MAX_TEXT_TOKENS),
    }
