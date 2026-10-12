# 本程序及代码是在人工智能工具辅助下完成的
# 人工智能工具名称：DeepSeek，型号：DeepSeek V4.1 Flash，开发公司：杭州深度求索人工智能基础技术研究有限公司，版本颁布日期：2026-09-10

"""附件 2（CMU-MOSEI 标准化特征）对齐版 / 未对齐版统一读取接口。

设计目标
--------
1. 用同一套接口读取 ``aligned_50.pkl`` 与 ``unaligned_50.pkl``，通过 ``version`` 参数切换。
2. 返回稳定样本 ID、三模态输入、分类与强度标签，以及**结构有效掩码**和**观测掩码**。
3. 保留原始序列位置（aligned 50 位、unaligned 音视频 500 位），不压缩内部零值位置。

掩码语义（本模块的明确定义，均已在附件 2 训练/验证数据上核实）
--------------------------------------------------------------
``*_struct_mask``
    该位置是否属于样本的**实际使用结构区间**。
    - 文本：``text_bert`` 第 1 路注意力掩码（已核实为连续前缀）。
    - aligned 音频/视觉：与文本位置一一对应，取 ``position < attn_len``
      （已核实 ``[attn_len, 50)`` 对两个模态恒为全零行）。
    - unaligned 音频：取 ``position < audio_lengths``（已核实该字段与实际非零行数
      100% 一致，其后恒为全零，无需扩展）。
    - unaligned 视觉：取**保守读取规则**（见下）。

``*_observed_mask``
    结构有效区间内该位置是否**实际获得可靠信息**，即 ``struct_mask & 该位置非全零``。
    结构有效区间内的全零行只标记为“未观测”，**不断言其成因**
    （可能是真实局部缺失，也可能是天然未检测到人脸等，字段无法区分）。

``text_struct_mask`` / ``text_content_mask`` / ``text_observed_mask``
    - ``text_struct_mask``：注意力有效区间（含特殊词元）。
    - ``text_content_mask``：注意力有效且**非特殊词元**（``[PAD]=0``、``[CLS]=101``、``[SEP]=102``）。
      已核实两版数据中首位恒为 ``[CLS]``、末有效位恒为 ``[SEP]``、段 ID 恒为 0。
    - ``text_observed_mask``：**只覆盖内容词元**，当前等于 ``text_content_mask``。
      注意：现有字段**无法识别“词元与注意力掩码均未改变的文本遮挡”**，
      因此该掩码不表示“文本内容一定完整可用”，只表示“该位置是有效内容词元”。
    - aligned 版中两个特殊词元位置对应的音频/视觉行恒为全零，属于“结构存在但无信号”。

未对齐版视觉的保守读取规则
--------------------------
``unaligned_50.pkl`` 的 ``vision_lengths`` 与数据不自洽：train 618/3395、valid 141/728
条样本在 ``[vision_lengths, 500)`` 内仍有非零行（最多 445 行），另有 30 条 train 样本在
声明长度内存在全零行；而 ``audio_lengths`` 100% 自洽。

本模块对视觉取：

    struct_len = min(500, max(vision_lengths, 最后一个非零视觉位置 + 1))

即保留声明区间**及其后已经存在的非零特征**，内部零行仍标为“未观测”。
这是**依据附件 2 训练/验证数据制定的保守读取规则**，用于避免静默丢弃已存在的特征；
它**不声称已经查明 ``vision_lengths`` 的原始语义**，也不改变原始数据与原始字段值。

原始声明长度与实际结构长度同时保留：``*_declared_lengths`` 与 ``*_struct_lengths``。
越界统计一律以**原始声明长度**为基准（``*_nonzero_beyond_declared``），
并单独给出被实际掩码保留下来的越界非零行数（``*_nonzero_rescued``），
不因结构长度扩展而清零。
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "attachment2"

VERSION_FILES = {
    "aligned": "aligned_50.pkl",
    "unaligned": "unaligned_50.pkl",
}
SPLITS = ("train", "valid", "test")

PAD_ID, CLS_ID, SEP_ID = 0, 101, 102
SPECIAL_TOKEN_IDS = (PAD_ID, CLS_ID, SEP_ID)

ID_SEPARATOR = "$_$"


@dataclass
class SplitData:
    """一个 ``(version, split)`` 下的全部样本，字段均为按样本第一维的 numpy 数组。"""

    version: str
    split: str

    sample_id: np.ndarray            # (N,) 稳定样本 ID，格式 video_id$_$clip_id
    video_id: np.ndarray             # (N,) ID 前半段
    clip_id: np.ndarray              # (N,) ID 后半段
    raw_text: np.ndarray             # (N,) 原始英文转写（仅阅读核查用）

    text_tokens: np.ndarray          # (N, 50) 词元编号
    text_segment: np.ndarray         # (N, 50) 分段编号（已核实恒为 0）
    text: np.ndarray                 # (N, 50, 768) 预计算文本表示
    text_struct_mask: np.ndarray     # (N, 50) 注意力有效区间（含特殊词元）
    text_content_mask: np.ndarray    # (N, 50) 注意力有效且非特殊词元
    text_observed_mask: np.ndarray   # (N, 50) 仅内容词元（= content_mask）

    audio: np.ndarray                # (N, Ta, 74)
    vision: np.ndarray               # (N, Tv, 35)
    audio_struct_mask: np.ndarray    # (N, Ta) 实际使用结构区间
    vision_struct_mask: np.ndarray   # (N, Tv) 实际使用结构区间
    audio_observed_mask: np.ndarray  # (N, Ta) struct & 非全零行
    vision_observed_mask: np.ndarray # (N, Tv) struct & 非全零行

    classification_labels: np.ndarray  # (N,) 0=负向 1=中性 2=正向
    regression_labels: np.ndarray      # (N,) 连续强度 [-3, 3]

    # 长度：声明长度（原始字段；aligned 下等于文本注意力长度）与实际结构长度
    audio_declared_lengths: np.ndarray   # (N,)
    vision_declared_lengths: np.ndarray  # (N,)
    audio_struct_lengths: np.ndarray     # (N,)
    vision_struct_lengths: np.ndarray    # (N,)

    # 越界统计：一律以原始声明长度为基准；rescued 为被实际掩码保留下来的越界非零行
    audio_nonzero_beyond_declared: np.ndarray   # (N,)
    vision_nonzero_beyond_declared: np.ndarray  # (N,)
    audio_nonzero_rescued: np.ndarray           # (N,)
    vision_nonzero_rescued: np.ndarray          # (N,)

    text_lengths: np.ndarray         # (N,) 文本注意力有效长度
    audio_lengths: Optional[np.ndarray] = None   # (N,) 仅未对齐版提供的原始字段
    vision_lengths: Optional[np.ndarray] = None  # (N,) 仅未对齐版提供的原始字段

    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.sample_id.shape[0])

    def modality_summary(self) -> dict:
        """结构有效区间内的零行比例等汇总指标。"""

        out = {"n": len(self), "version": self.version, "split": self.split}
        for name in ("audio", "vision"):
            struct = np.asarray(getattr(self, f"{name}_struct_mask")).reshape(-1)
            observed = np.asarray(getattr(self, f"{name}_observed_mask")).reshape(-1)
            out[f"{name}_struct_positions"] = int(struct.sum())
            out[f"{name}_observed_positions"] = int(observed.sum())
            out[f"{name}_zero_row_ratio_in_struct"] = (
                float(1.0 - observed.sum() / struct.sum()) if struct.sum() else 0.0
            )
        ts = np.asarray(self.text_struct_mask).reshape(-1)
        tc = np.asarray(self.text_content_mask).reshape(-1)
        out["text_struct_positions"] = int(ts.sum())
        out["text_content_positions"] = int(tc.sum())
        out["text_special_positions"] = int(ts.sum() - tc.sum())
        return out


def _row_nonzero(signal: np.ndarray) -> np.ndarray:
    """(N, T, D) -> (N, T) 该位置是否为非全零行。"""
    return np.asarray(signal != 0).any(axis=-1)


def _last_nonzero_plus_one(nonzero_rows: np.ndarray) -> np.ndarray:
    """(N, T) 非零行掩码 -> (N,) 最后一个非零位置 + 1（无任何非零行时为 0）。"""
    nonzero_rows = np.asarray(nonzero_rows)
    has = nonzero_rows.any(axis=1)
    # 反转后 argmax 找到原序列中最后一个 True 的偏移
    offset = np.argmax(nonzero_rows[:, ::-1], axis=1)
    last_plus_one = nonzero_rows.shape[1] - offset
    return np.where(has, last_plus_one, 0).astype(np.int64)


def _parse_ids(sample_id: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    parts = [str(s).split(ID_SEPARATOR, 1) for s in sample_id]
    video = np.asarray([p[0] for p in parts], dtype=object)
    clip = np.asarray([p[1] if len(p) > 1 else "" for p in parts], dtype=object)
    return video, clip


def _split_from_fields(fields: dict, version: str, split: str) -> SplitData:
    tokens = np.asarray(fields["text_bert"])
    text_tokens = np.ascontiguousarray(tokens[:, 0, :].astype(np.int64, copy=False))
    attn = tokens[:, 1, :] != 0
    seg = np.ascontiguousarray(tokens[:, 2, :].astype(np.int64, copy=False))

    text_struct = np.ascontiguousarray(attn)
    text_content = np.ascontiguousarray(attn & ~np.isin(text_tokens, SPECIAL_TOKEN_IDS))
    # 文本观测掩码只覆盖内容词元；字段无法识别“词元与掩码均未变的文本遮挡”。
    text_observed = text_content.copy()

    audio = np.asarray(fields["audio"])
    vision = np.asarray(fields["vision"])
    text = np.asarray(fields["text"])

    audio_nonzero = _row_nonzero(audio)
    vision_nonzero = _row_nonzero(vision)

    text_lengths = text_struct.sum(axis=1).astype(np.int64)
    has_lengths = "audio_lengths" in fields and "vision_lengths" in fields

    audio_positions = audio.shape[1]
    vision_positions = vision.shape[1]
    ar_audio = np.arange(audio_positions)[None, :]
    ar_vision = np.arange(vision_positions)[None, :]

    if version == "aligned":
        # 对齐版：三模态共用同一批序列位置，声明长度与结构长度都取文本注意力长度。
        audio_declared = text_lengths.copy()
        vision_declared = text_lengths.copy()
        audio_struct_len = audio_declared.copy()
        vision_struct_len = vision_declared.copy()
        audio_lengths = None
        vision_lengths = None
    elif version == "unaligned":
        if not has_lengths:
            raise KeyError("unaligned 版缺少 audio_lengths / vision_lengths 字段")
        audio_lengths = np.asarray(fields["audio_lengths"]).astype(np.int64)
        vision_lengths = np.asarray(fields["vision_lengths"]).astype(np.int64)
        audio_declared = audio_lengths.copy()
        vision_declared = vision_lengths.copy()
        # 音频：已核实字段与实际非零行数 100% 一致，无需扩展。
        audio_struct_len = audio_declared.copy()
        # 视觉：保守读取规则，保留声明区间及其后已有的非零特征（上限为位置总数）。
        vision_struct_len = np.clip(
            np.maximum(vision_declared, _last_nonzero_plus_one(vision_nonzero)),
            0,
            vision_positions,
        ).astype(np.int64)
    else:
        raise ValueError(f"未知 version: {version!r}，可选 {sorted(VERSION_FILES)}")

    audio_declared_mask = np.ascontiguousarray(ar_audio < audio_declared[:, None])
    vision_declared_mask = np.ascontiguousarray(ar_vision < vision_declared[:, None])
    audio_struct = np.ascontiguousarray(ar_audio < audio_struct_len[:, None])
    vision_struct = np.ascontiguousarray(ar_vision < vision_struct_len[:, None])

    # 越界统计一律以原始声明长度为基准
    audio_beyond = (audio_nonzero & ~audio_declared_mask).sum(axis=1).astype(np.int64)
    vision_beyond = (vision_nonzero & ~vision_declared_mask).sum(axis=1).astype(np.int64)
    audio_rescued = (audio_nonzero & ~audio_declared_mask & audio_struct).sum(axis=1).astype(np.int64)
    vision_rescued = (vision_nonzero & ~vision_declared_mask & vision_struct).sum(axis=1).astype(np.int64)

    sample_id = np.asarray(fields["id"])
    video_id, clip_id = _parse_ids(sample_id)

    meta = {
        "version": version,
        "split": split,
        "n": int(sample_id.shape[0]),
        "text_positions": int(text_tokens.shape[1]),
        "audio_positions": int(audio_positions),
        "vision_positions": int(vision_positions),
        "audio_dim": int(audio.shape[2]),
        "vision_dim": int(vision.shape[2]),
        "text_dim": int(text.shape[2]),
        "has_length_fields": bool(has_lengths),
        "struct_rule": {
            "text": "text_bert attention mask (contiguous prefix)",
            "audio": (
                "position < text_attention_length"
                if version == "aligned"
                else "position < audio_lengths (verified self-consistent; no expansion)"
            ),
            "vision": (
                "position < text_attention_length"
                if version == "aligned"
                else "position < min(500, max(vision_lengths, last_nonzero_position + 1))"
            ),
            "observed": "struct_mask & row is not all-zero; zero rows inside struct are 'unobserved'",
            "text_observed": "content tokens only (attention & not [PAD]/[CLS]/[SEP])",
        },
        "vision_conservative_rule": (
            None
            if version == "aligned"
            else "vision struct length keeps declared range plus any existing nonzero rows beyond it; "
            "conservative reading rule derived from attachment-2 train/valid data only; "
            "the original semantics of vision_lengths is NOT claimed to be resolved"
        ),
        "beyond_declared_basis": "original declared length field (not the expanded struct length)",
        "labels": {
            "classification": "0=negative,1=neutral,2=positive",
            "regression": "continuous intensity in [-3,3]",
        },
        "dtype": {
            "text": str(text.dtype),
            "audio": str(audio.dtype),
            "vision": str(vision.dtype),
            "text_tokens": str(text_tokens.dtype),
        },
    }

    return SplitData(
        version=version,
        split=split,
        sample_id=sample_id,
        video_id=video_id,
        clip_id=clip_id,
        raw_text=np.asarray(fields["raw_text"]),
        text_tokens=text_tokens,
        text_segment=seg,
        text=text,
        text_struct_mask=text_struct,
        text_content_mask=text_content,
        text_observed_mask=text_observed,
        audio=audio,
        vision=vision,
        audio_struct_mask=audio_struct,
        vision_struct_mask=vision_struct,
        audio_observed_mask=np.ascontiguousarray(audio_nonzero & audio_struct),
        vision_observed_mask=np.ascontiguousarray(vision_nonzero & vision_struct),
        classification_labels=np.asarray(fields["classification_labels"]).reshape(-1),
        regression_labels=np.asarray(fields["regression_labels"], dtype=np.float64).reshape(-1),
        audio_declared_lengths=audio_declared,
        vision_declared_lengths=vision_declared,
        audio_struct_lengths=audio_struct_len,
        vision_struct_lengths=vision_struct_len,
        audio_nonzero_beyond_declared=audio_beyond,
        vision_nonzero_beyond_declared=vision_beyond,
        audio_nonzero_rescued=audio_rescued,
        vision_nonzero_rescued=vision_rescued,
        text_lengths=text_lengths,
        audio_lengths=audio_lengths,
        vision_lengths=vision_lengths,
        meta=meta,
    )


class Attachment2:
    """附件 2 某一版本的惰性读取器，pickle 只加载一次，各划分按需切分并缓存。"""

    def __init__(self, version: str = "aligned"):
        if version not in VERSION_FILES:
            raise ValueError(f"未知 version: {version!r}，可选 {sorted(VERSION_FILES)}")
        self.version = version
        self.path = DATA_DIR / VERSION_FILES[version]
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        self._raw: Optional[dict] = None
        self._cache: Dict[str, SplitData] = {}

    def _load(self) -> dict:
        if self._raw is None:
            with self.path.open("rb") as handle:
                self._raw = pickle.load(handle)
        return self._raw

    def splits(self) -> tuple:
        return tuple(self._load().keys())

    def raw_fields(self, name: str) -> dict:
        """返回该划分的原始字段字典（未加工），用于核对数值与位置未被改写。"""

        raw = self._load()
        if name not in raw:
            raise KeyError(f"{self.path.name} 中无划分 {name!r}，可选 {sorted(raw)}")
        return raw[name]

    def split(self, name: str) -> SplitData:
        if name not in self._cache:
            raw = self._load()
            if name not in raw:
                raise KeyError(f"{self.path.name} 中无划分 {name!r}，可选 {sorted(raw)}")
            self._cache[name] = _split_from_fields(raw[name], self.version, name)
        return self._cache[name]

    def __repr__(self) -> str:  # pragma: no cover - 便于交互排查
        return f"Attachment2(version={self.version!r}, path={self.path.name})"


def load_split(version: str = "aligned", split: str = "train") -> SplitData:
    """读取指定版本与划分。

    参数
    ----
    version : ``"aligned"`` 或 ``"unaligned"``
    split   : ``"train"`` / ``"valid"`` / ``"test"``

    返回
    ----
    :class:`SplitData`
    """

    return Attachment2(version).split(split)


def load_splits(version: str = "aligned", splits: Iterable[str] = ("train", "valid")) -> Dict[str, SplitData]:
    """一次读取同一版本的多个划分，共享同一次 pickle 加载。"""

    reader = Attachment2(version)
    return {name: reader.split(name) for name in splits}
