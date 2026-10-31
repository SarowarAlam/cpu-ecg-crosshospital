"""Handcrafted ECG features for the gradient-boosting baseline: per-lead
statistical/spectral features plus R-peak/heart-rate-variability features
from a Pan-Tompkins-style detector, computed with scipy only (no extra
ECG-specific dependency). Operates on the signal at its native sampling
rate (500 Hz for these datasets), independent of the separate downsampling
used for the CNN baseline.
"""
from __future__ import annotations

import numpy as np
from scipy import signal as sg
from scipy import stats as st


def detect_r_peaks(lead_ii: np.ndarray, fs: float) -> np.ndarray:
    """Pan-Tompkins-style R-peak detector. Returns sample indices of detected peaks."""
    nyq = fs / 2.0
    b, a = sg.butter(2, [5 / nyq, 15 / nyq], btype="band")
    filtered = sg.filtfilt(b, a, lead_ii)
    derivative = np.diff(filtered, prepend=filtered[0])
    squared = derivative ** 2
    win = max(1, int(0.15 * fs))
    integrated = np.convolve(squared, np.ones(win) / win, mode="same")

    if integrated.max() <= 0:
        return np.array([], dtype=int)
    min_distance = max(1, int(0.3 * fs))  # 300 ms refractory period, ~200 bpm cap
    height = 0.3 * integrated.max()
    peaks, _ = sg.find_peaks(integrated, height=height, distance=min_distance)
    return peaks


def hrv_features(r_peaks: np.ndarray, fs: float) -> dict[str, float]:
    if len(r_peaks) < 3:
        return {
            "hr_mean": np.nan, "rr_mean": np.nan, "rr_std": np.nan,
            "rr_min": np.nan, "rr_max": np.nan, "rmssd": np.nan,
            "pnn50": np.nan, "n_beats": float(len(r_peaks)),
        }
    rr = np.diff(r_peaks) / fs  # seconds
    rr_diff = np.diff(rr)
    return {
        "hr_mean": 60.0 / np.mean(rr),
        "rr_mean": float(np.mean(rr)),
        "rr_std": float(np.std(rr)),
        "rr_min": float(np.min(rr)),
        "rr_max": float(np.max(rr)),
        "rmssd": float(np.sqrt(np.mean(rr_diff ** 2))) if len(rr_diff) else np.nan,
        "pnn50": float(np.mean(np.abs(rr_diff) > 0.05)) if len(rr_diff) else np.nan,
        "n_beats": float(len(r_peaks)),
    }


def per_lead_features(lead_signal: np.ndarray, fs: float, prefix: str) -> dict[str, float]:
    x = lead_signal.astype(np.float64)
    feats = {
        f"{prefix}_mean": float(np.mean(x)),
        f"{prefix}_std": float(np.std(x)),
        f"{prefix}_skew": float(st.skew(x)),
        f"{prefix}_kurtosis": float(st.kurtosis(x)),
        f"{prefix}_min": float(np.min(x)),
        f"{prefix}_max": float(np.max(x)),
        f"{prefix}_ptp": float(np.ptp(x)),
        f"{prefix}_rms": float(np.sqrt(np.mean(x ** 2))),
        f"{prefix}_zcr": float(np.mean(np.diff(np.sign(x)) != 0)),
    }
    freqs = np.fft.rfftfreq(len(x), d=1.0 / fs)
    power = np.abs(np.fft.rfft(x)) ** 2
    if power.sum() > 0:
        feats[f"{prefix}_dom_freq"] = float(freqs[np.argmax(power[1:]) + 1]) if len(power) > 1 else 0.0
        p_norm = power / power.sum()
        p_norm = p_norm[p_norm > 0]
        feats[f"{prefix}_spec_entropy"] = float(-np.sum(p_norm * np.log(p_norm)))
    else:
        feats[f"{prefix}_dom_freq"] = 0.0
        feats[f"{prefix}_spec_entropy"] = 0.0
    return feats


def extract_features(signal_3lead: np.ndarray, fs: float, lead_names: list[str]) -> dict[str, float]:
    """signal_3lead: (3, n_samples), lead order matching lead_names (e.g. ["I", "II", "V1"])."""
    feats: dict[str, float] = {}
    for i, name in enumerate(lead_names):
        feats.update(per_lead_features(signal_3lead[i], fs, prefix=f"lead_{name}"))

    ii_idx = lead_names.index("II") if "II" in lead_names else 1
    r_peaks = detect_r_peaks(signal_3lead[ii_idx], fs)
    feats.update(hrv_features(r_peaks, fs))
    return feats
