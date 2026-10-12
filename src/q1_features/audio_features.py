# 本程序及代码是在人工智能工具辅助下完成的
# 人工智能工具名称：DeepSeek，型号：DeepSeek V4.1 Flash，开发公司：杭州深度求索人工智能基础技术研究有限公司，版本颁布日期：2026-09-10

"""16 kHz音频的17维原生短窗特征。"""

from __future__ import annotations

import numpy as np

from scipy.fftpack import dct

SAMPLE_RATE = 16000

WINDOW_SAMPLES = 400

HOP_SAMPLES = 160

N_FFT = 512

N_MEL_FILTERS = 26

N_MFCC = 13

F0_MIN_HZ = 60.0

F0_MAX_HZ = 400.0

F0_ACF_PEAK_MIN = 0.30

F0_AMPLITUDE_FLOOR_LOG_RMS = float(np.log(1e-3))

EPS = 1e-10

FILL_VALUE = 0.0

FEATURE_NAMES = (
    [f"mfcc_{i}" for i in range(N_MFCC)]
    + ["log_rms", "f0_hz", "spectral_centroid_hz", "zero_crossing_rate"]
)

N_FEATURES = len(FEATURE_NAMES)

def hz_to_mel(freq: np.ndarray | float):
    return 2595.0 * np.log10(1.0 + np.asarray(freq, dtype=np.float64) / 700.0)

def mel_to_hz(mel: np.ndarray | float):
    return 700.0 * (10.0 ** (np.asarray(mel, dtype=np.float64) / 2595.0) - 1.0)

def mel_filterbank(n_filters: int = N_MEL_FILTERS, n_fft: int = N_FFT,
                   sample_rate: int = SAMPLE_RATE, fmin: float = 0.0,
                   fmax: float | None = None) -> np.ndarray:
    """单位峰值三角 mel 滤波器组，形状 (n_filters, n_fft//2+1)。"""

    fmax = sample_rate / 2 if fmax is None else fmax
    mel_edges = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_filters + 2)
    hz_edges = mel_to_hz(mel_edges)
    n_bins = n_fft // 2 + 1
    bins = np.floor((n_fft + 1) * hz_edges / sample_rate).astype(int)
    bins = np.clip(bins, 0, n_bins - 1)
    bank = np.zeros((n_filters, n_bins), dtype=np.float64)
    for i in range(n_filters):
        left, center, right = int(bins[i]), int(bins[i + 1]), int(bins[i + 2])
        if center <= left:
            center = min(left + 1, n_bins - 1)
        if right <= center:
            right = min(center + 1, n_bins - 1)
        for k in range(left, center):
            bank[i, k] = (k - left) / max(1, center - left)
        for k in range(center, right):
            bank[i, k] = (right - k) / max(1, right - center)
    return bank

def hamming(n: int = WINDOW_SAMPLES) -> np.ndarray:
    return np.hamming(n)

def window_features(signal: np.ndarray, bank: np.ndarray, win: np.ndarray) -> dict:
    """返回 (n_windows, 17) 特征、基频标记与逐窗诊断。"""

    n = signal.shape[0]
    if n < WINDOW_SAMPLES:
        raise ValueError(f"音频仅 {n} 个样本，不足一个 25 ms 窗")
    n_windows = (n - WINDOW_SAMPLES) // HOP_SAMPLES + 1
    bank_t = bank.T  # (n_bins, n_filters)

    features = np.zeros((n_windows, N_FEATURES), dtype=np.float64)
    sample_start = np.zeros(n_windows, dtype=np.int64)
    sample_end = np.zeros(n_windows, dtype=np.int64)
    pitch_observed = np.zeros(n_windows, dtype=bool)
    pitch_reason = np.empty(n_windows, dtype=object)
    f0_at_boundary = np.zeros(n_windows, dtype=bool)
    log_rms_all = np.zeros(n_windows, dtype=np.float64)
    acf_peak_all = np.zeros(n_windows, dtype=np.float64)

    lag_min = int(np.ceil(SAMPLE_RATE / F0_MAX_HZ))
    lag_max = int(np.floor(SAMPLE_RATE / F0_MIN_HZ))
    lag_max = min(lag_max, WINDOW_SAMPLES - 1)

    for i in range(n_windows):
        start = i * HOP_SAMPLES
        stop = start + WINDOW_SAMPLES
        raw = signal[start:stop].astype(np.float64)
        sample_start[i] = start
        sample_end[i] = stop

        centered = raw - raw.mean()
        # 对数 RMS：去均值、未加窗
        rms = float(np.sqrt(np.mean(centered ** 2)))
        log_rms = float(np.log(max(rms, EPS)))
        log_rms_all[i] = log_rms

        # 过零率：去均值、未加窗
        signs = np.signbit(centered)
        zcr = float(np.count_nonzero(signs[1:] != signs[:-1])) / (WINDOW_SAMPLES - 1)

        tapered = centered * win
        spectrum = np.fft.rfft(tapered, n=N_FFT)
        power = (np.abs(spectrum) ** 2)
        magnitude = np.abs(spectrum)

        mel_energy = power @ bank_t  # (n_filters,)
        log_mel = np.log(np.maximum(mel_energy, EPS))
        mfcc = dct(log_mel, type=2, norm="ortho")[:N_MFCC]

        freqs = np.fft.rfftfreq(N_FFT, d=1.0 / SAMPLE_RATE)
        total_mag = float(magnitude.sum())
        centroid = float((freqs * magnitude).sum() / total_mag) if total_mag > 0 else FILL_VALUE

        # 基频：FFT 自相关
        acf = np.fft.irfft(np.abs(np.fft.rfft(tapered, n=N_FFT)) ** 2, n=N_FFT)
        if acf[0] > 0:
            acf = acf / acf[0]
        seg = acf[lag_min : lag_max + 1]
        if seg.size:
            best = int(np.argmax(seg))
            peak = float(seg[best])
            lag = lag_min + best
        else:
            peak, lag = 0.0, 0
        acf_peak_all[i] = peak
        if log_rms < F0_AMPLITUDE_FLOOR_LOG_RMS:
            pitch_observed[i] = False
            pitch_reason[i] = "below_amplitude_floor"
            f0 = FILL_VALUE
        elif peak < F0_ACF_PEAK_MIN or lag <= 0:
            pitch_observed[i] = False
            pitch_reason[i] = "weak_periodicity"
            f0 = FILL_VALUE
        else:
            pitch_observed[i] = True
            # 落在搜索区间端点的 lag 只是“下界/上界”，不是精确估计，单独标记
            at_boundary = lag <= lag_min or lag >= lag_max
            f0_at_boundary[i] = at_boundary
            pitch_reason[i] = "observed_at_search_boundary" if at_boundary else "observed"
            f0 = SAMPLE_RATE / lag

        features[i, :N_MFCC] = mfcc
        features[i, N_MFCC] = log_rms
        features[i, N_MFCC + 1] = f0
        features[i, N_MFCC + 2] = centroid
        features[i, N_MFCC + 3] = zcr

    tail_start = WINDOW_SAMPLES + (n_windows - 1) * HOP_SAMPLES
    return {
        "features": features,
        "sample_start": sample_start,
        "sample_end": sample_end,
        "pitch_observed": pitch_observed,
        "pitch_reason": pitch_reason,
        "f0_at_boundary": f0_at_boundary,
        "log_rms": log_rms_all,
        "acf_peak": acf_peak_all,
        "n_windows": int(n_windows),
        "tail_dropped_samples": int(n - tail_start),
        "tail_dropped_start_sample": int(tail_start),
        "covered_samples": int(tail_start),
    }

FILL_VALUE = 0.0
