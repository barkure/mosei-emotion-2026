# 本程序及代码是在人工智能工具辅助下完成的
# 人工智能工具名称：DeepSeek，型号：DeepSeek V4.1 Flash，开发公司：杭州深度求索人工智能基础技术研究有限公司，版本颁布日期：2026-09-10

"""统一读取接口的验证脚本：两版各取一批样本，核对掩码长度、张量形状与 ID 对应。

用法::

    pixi run verify-data-io

脚本只读取附件 2 的 train / valid，不训练模型、不写任何数据文件。
"""

from __future__ import annotations
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
for _p in (str(HERE.parent), str(HERE)):          # src/ 与本文件所在目录
    if _p not in sys.path:
        sys.path.insert(0, _p)


import numpy as np

from data_io import Attachment2

FAILURES: list[str] = []


def check(condition: bool, message: str) -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {message}")
    if not condition:
        FAILURES.append(message)


def verify_version(reader: Attachment2, split: str, batch: int) -> None:
    version = reader.version
    print(f"\n=== {version} / {split} ===")
    data = reader.split(split)
    n = len(data)

    print(
        f"  样本数={n}  text={data.text.shape}  audio={data.audio.shape}  "
        f"vision={data.vision.shape}  text_positions={data.meta['text_positions']}"
    )

    # ---- 形状 ----
    check(data.text.shape == (n, 50, 768), "text 形状为 (N, 50, 768)")
    check(data.audio.shape == (n, data.audio.shape[1], 74), "audio 末维为 74")
    check(data.vision.shape == (n, data.vision.shape[1], 35), "vision 末维为 35")
    check(
        data.text_struct_mask.shape == data.text_tokens.shape == (n, 50),
        "文本掩码与词元形状一致 (N, 50)",
    )
    check(data.audio_struct_mask.shape == data.audio.shape[:2], "audio 掩码形状与音频一致")
    check(data.vision_struct_mask.shape == data.vision.shape[:2], "vision 掩码形状与视觉一致")
    expected_positions = 50 if version == "aligned" else 500
    check(data.audio.shape[1] == expected_positions, f"{version} 音频位置数 = {expected_positions}")
    check(data.vision.shape[1] == expected_positions, f"{version} 视觉位置数 = {expected_positions}")
    check(data.text.shape[1] == 50, "文本位置数恒为 50（两版一致）")

    # ---- 原始位置未被压缩 ----
    check(
        data.audio_struct_mask.sum(axis=1).max() <= data.audio.shape[1],
        "结构掩码不超出原始位置范围（未压缩内部零值位置）",
    )

    # ---- 掩码长度 ----
    attn_len = data.text_struct_mask.sum(axis=1)
    check(np.array_equal(attn_len, data.text_lengths), "文本结构长度 == 注意力掩码行和")
    check(bool((attn_len >= 3).all() and (attn_len <= 50).all()), "文本长度落在 [3, 50]")

    a_len = data.audio_struct_mask.sum(axis=1)
    v_len = data.vision_struct_mask.sum(axis=1)
    check(np.array_equal(a_len, data.audio_struct_lengths), "音频结构掩码行和 == audio_struct_lengths")
    check(np.array_equal(v_len, data.vision_struct_lengths), "视觉结构掩码行和 == vision_struct_lengths")
    if version == "aligned":
        check(np.array_equal(a_len, attn_len), "aligned: 音频结构长度 == 文本注意力长度")
        check(np.array_equal(v_len, attn_len), "aligned: 视觉结构长度 == 文本注意力长度")
        check(
            np.array_equal(data.audio_declared_lengths, data.audio_struct_lengths)
            and np.array_equal(data.vision_declared_lengths, data.vision_struct_lengths),
            "aligned: 声明长度 == 结构长度",
        )
    else:
        check(np.array_equal(a_len, data.audio_lengths), "unaligned: 音频结构长度 == audio_lengths")
        check(
            np.array_equal(data.audio_struct_lengths, data.audio_declared_lengths),
            "unaligned: 音频结构长度未被扩展（字段已自洽）",
        )
        check(
            bool((data.vision_struct_lengths >= data.vision_declared_lengths).all()),
            "unaligned: 视觉结构长度 >= 声明长度（保守规则不缩短区间）",
        )
        check(
            bool((data.vision_struct_lengths <= data.vision.shape[1]).all()),
            "unaligned: 视觉结构长度不超过位置总数 500",
        )
        # 结构长度 = max(声明, 最后一个非零位置+1)
        nz = np.asarray(data.vision != 0).any(axis=-1)
        has = nz.any(axis=1)
        last_plus_one = np.where(has, nz.shape[1] - np.argmax(nz[:, ::-1], axis=1), 0)
        expected = np.clip(np.maximum(data.vision_declared_lengths, last_plus_one), 0, nz.shape[1])
        check(
            np.array_equal(data.vision_struct_lengths, expected),
            "unaligned: 视觉结构长度 == min(500, max(vision_lengths, 最后一个非零位置+1))",
        )

    # ---- 观测掩码 ----
    check(
        bool((data.audio_observed_mask & ~data.audio_struct_mask).sum() == 0),
        "audio observed ⊆ struct",
    )
    check(
        bool((data.vision_observed_mask & ~data.vision_struct_mask).sum() == 0),
        "vision observed ⊆ struct",
    )
    check(
        bool((data.text_content_mask & ~data.text_struct_mask).sum() == 0),
        "text content ⊆ attention",
    )
    # text_observed_mask 只覆盖内容词元
    check(
        np.array_equal(data.text_observed_mask, data.text_content_mask),
        "text_observed_mask 只覆盖内容词元（== text_content_mask）",
    )
    check(
        bool((data.text_observed_mask & ~data.text_struct_mask).sum() == 0),
        "text observed ⊆ attention（单独保留 text_struct_mask）",
    )
    check(
        not np.array_equal(data.text_observed_mask, data.text_struct_mask),
        "text_observed_mask 与 text_struct_mask 是两个不同的掩码",
    )
    # [CLS] + [SEP] 恰好 2 个特殊位置
    special = data.text_struct_mask.sum(axis=1) - data.text_content_mask.sum(axis=1)
    check(bool((special == 2).all()), "每条样本恰有 2 个特殊词元位置（[CLS] 与 [SEP]）")
    check(
        bool(
            (data.text_tokens[:, 0] == 101).all()
            and (data.text_tokens[np.arange(n), attn_len - 1] == 102).all()
        ),
        "[CLS]=101 位于首位、[SEP]=102 位于末有效位",
    )

    # ---- 越界统计以声明长度为基准 ----
    if version == "unaligned":
        declared_mask = np.arange(data.vision.shape[1])[None, :] < data.vision_declared_lengths[:, None]
        nz_rows = np.asarray(data.vision != 0).any(axis=-1)
        expected_beyond = (nz_rows & ~declared_mask).sum(axis=1)
        check(
            np.array_equal(data.vision_nonzero_beyond_declared, expected_beyond),
            "unaligned: 视觉越界统计以原始声明长度为基准",
        )
        rescued = (nz_rows & ~declared_mask & data.vision_struct_mask).sum(axis=1)
        check(
            np.array_equal(data.vision_nonzero_rescued, rescued),
            "unaligned: rescued 计数 == 声明长度之外且落在结构掩码内的非零行数",
        )
        check(
            bool((data.vision_nonzero_rescued <= data.vision_nonzero_beyond_declared).all()),
            "unaligned: rescued <= beyond（扩展不清零异常计数）",
        )
        check(
            int(data.vision_nonzero_beyond_declared.sum()) > 0,
            "unaligned: 确实存在越界非零行（异常未被清零）",
        )

    # ---- 结构区间内的零行比例 ----
    zero_a = 1.0 - data.audio_observed_mask.sum() / max(1, data.audio_struct_mask.sum())
    zero_v = 1.0 - data.vision_observed_mask.sum() / max(1, data.vision_struct_mask.sum())
    print(f"  结构区间内零行比例: audio={zero_a:.4f}  vision={zero_v:.4f}")
    check(0.0 <= zero_a <= 1.0 and 0.0 <= zero_v <= 1.0, "零行比例落在 [0, 1]")

    # ---- 标签（精确取值检查，不做整数转换） ----
    cls = np.asarray(data.classification_labels, dtype=np.float64)
    reg = data.regression_labels
    unique_cls = sorted(set(cls.tolist()))
    check(
        set(unique_cls) <= {0.0, 1.0, 2.0},
        f"分类标签精确取值 ⊆ {{0,1,2}}（实际取值 {unique_cls}）",
    )
    check(bool((reg >= -3).all() and (reg <= 3).all()), "强度标签落在 [-3, 3]")
    check(
        bool((np.sign(reg).astype(int) + 1 == cls).all()),
        "分类标签 == 强度符号 + 1（全部样本一致）",
    )

    # ---- 取一批样本 ----
    idx = np.linspace(0, n - 1, batch).round().astype(int)
    print(f"\n  样本抽查（索引 {idx.tolist()}）：")
    print(
        f"    {'idx':>5} {'sample_id':>22} {'textL':>6} {'aDecl':>6} {'aLen':>6} "
        f"{'vDecl':>6} {'vLen':>6} {'vBeyond':>8} {'cls':>4} {'reg':>6}"
    )
    for i in idx:
        print(
            f"    {i:>5} {str(data.sample_id[i]):>22} {int(attn_len[i]):>6} "
            f"{int(data.audio_declared_lengths[i]):>6} {int(a_len[i]):>6} "
            f"{int(data.vision_declared_lengths[i]):>6} {int(v_len[i]):>6} "
            f"{int(data.vision_nonzero_beyond_declared[i]):>8} "
            f"{cls[i]:>4.0f} {reg[i]:>6.2f}"
        )

    ok = True
    for i in idx:
        ok &= int(data.audio_struct_mask[i].sum()) == int(a_len[i])
        ok &= int(data.vision_struct_mask[i].sum()) == int(v_len[i])
    check(ok, "抽查样本的结构掩码长度与观测掩码内部一致")

    del data


def verify_anomaly_sample(reader: Attachment2) -> None:
    """真实异常样本：声明长度之后的非零视觉位置仍位于实际结构掩码内，且原始位置与数值不变。"""

    print("\n=== 真实异常样本检查（unaligned / train，vision_lengths 越界非零） ===")
    data = reader.split("train")
    raw = reader.raw_fields("train")

    candidates = np.where(data.vision_nonzero_beyond_declared > 0)[0]
    check(len(candidates) > 0, f"存在越界非零视觉样本（{len(candidates)} 条）")
    if len(candidates) == 0:
        return

    i = int(candidates[0])
    positions = np.arange(data.vision.shape[1])
    declared = int(data.vision_declared_lengths[i])
    struct_len = int(data.vision_struct_lengths[i])
    nonzero_rows = np.asarray(data.vision[i] != 0).any(axis=-1)
    beyond_positions = np.where(nonzero_rows & (positions >= declared))[0]

    print(
        f"  样本 idx={i} id={data.sample_id[i]}  声明长度={declared}  "
        f"结构长度={struct_len}  越界非零行={int(data.vision_nonzero_beyond_declared[i])}"
    )
    print(f"  越界非零位置（前 8 个）: {beyond_positions[:8].tolist()}")

    check(len(beyond_positions) > 0, "该样本在声明长度之后存在非零视觉位置")
    p = int(beyond_positions[0])
    check(p >= declared, f"该位置 {p} 位于声明长度 {declared} 之后")
    check(p < struct_len, f"该位置 {p} 位于实际结构长度 {struct_len} 之内（被保留）")
    check(bool(data.vision_struct_mask[i, p]), "该位置在实际结构掩码内为 True")
    check(bool(data.vision_observed_mask[i, p]), "该位置在观测掩码内为 True（非全零行）")

    # 与原始 pickle 对照：位置未被平移、数值未被改写
    raw_row = np.asarray(raw["vision"][i])[p]
    check(np.array_equal(raw_row, data.vision[i, p]), "该位置数值与原始数据逐元素完全一致")
    check(bool((raw_row != 0).any()), "原始数据该位置确实非零")
    check(int(raw["vision_lengths"][i]) == declared, "声明长度与原始 vision_lengths 字段一致")
    check(
        np.array_equal(np.asarray(raw["vision"][i]), data.vision[i]),
        "整条样本的视觉张量与原始数据完全一致（位置与数值均未改动）",
    )
    # 声明区间内的零行仍标为未观测
    inside_zero = np.where(~nonzero_rows & (positions < declared))[0]
    if len(inside_zero) > 0:
        q = int(inside_zero[0])
        check(
            not bool(data.vision_observed_mask[i, q]),
            f"声明区间内的全零行（位置 {q}）仍标为未观测",
        )
    else:
        print("  （该样本声明区间内无全零行，跳过内部零行检查）")

    del data


def verify_cross_version(readers: dict) -> None:
    print("\n=== 跨版本 ID / 文本 / 标签一致性 ===")
    for split in ("train", "valid"):
        left = readers["aligned"].split(split)
        right = readers["unaligned"].split(split)
        same_ids = [str(x) for x in left.sample_id] == [str(x) for x in right.sample_id]
        check(len(left) == len(right), f"{split}: 两版样本数相同 ({len(left)})")
        check(same_ids, f"{split}: 两版 ID 顺序与取值完全对应")
        check(
            np.array_equal(left.classification_labels, right.classification_labels)
            and np.array_equal(left.regression_labels, right.regression_labels),
            f"{split}: 两版标签一致",
        )
        check(
            np.array_equal(left.text_struct_mask, right.text_struct_mask),
            f"{split}: 两版文本注意力掩码一致",
        )
        check(
            np.array_equal(left.text_lengths, right.text_lengths),
            f"{split}: 同一 ID 在两版中的文本长度一致",
        )


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print("统一读取接口验证（仅 train / valid，不训练、不写文件）")
    readers = {version: Attachment2(version) for version in ("aligned", "unaligned")}
    for version, reader in readers.items():
        for split in ("train", "valid"):
            verify_version(reader, split, batch=4)
    verify_anomaly_sample(readers["unaligned"])
    verify_cross_version(readers)

    print("\n" + "=" * 60)
    if FAILURES:
        print(f"结果：{len(FAILURES)} 项失败")
        for item in FAILURES:
            print(f"  - {item}")
        raise SystemExit(1)
    print("结果：全部检查通过")


if __name__ == "__main__":
    main()
