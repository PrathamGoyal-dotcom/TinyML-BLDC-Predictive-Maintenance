"""28-feature time- and frequency-domain extractor for BLDC vibration & current windows."""

from __future__ import annotations

from typing import Optional

import numpy as np

from app.schemas import FEATURE_NAMES, SIGNAL_FEATURE_SUFFIXES

_EPS = 1e-12


def _harmonic_amplitude(
    freqs: np.ndarray,
    spectrum: np.ndarray,
    target_hz: float,
    nyquist_hz: float,
    df: float,
) -> float:
    """Return peak FFT amplitude in a narrow band around target_hz."""
    if target_hz <= 0.0 or target_hz >= nyquist_hz or len(freqs) == 0:
        return 0.0
    tol = max(2.0, 0.12 * target_hz, 1.5 * df)
    mask = (freqs >= max(0.5, target_hz - tol)) & (freqs <= min(nyquist_hz, target_hz + tol))
    if not np.any(mask):
        idx = int(np.argmin(np.abs(freqs - target_hz)))
        return float(spectrum[idx])
    return float(np.max(spectrum[mask]))


def extract_signal_features(
    signal: list[float] | np.ndarray,
    *,
    sample_rate_hz: float,
    rpm: float,
    hf_bearing_min_hz: float = 250.0,
) -> dict[str, float]:
    """Extract 14 time- and frequency-domain features from a 1D sensor window."""
    x = np.asarray(signal, dtype=np.float64)
    n = len(x)
    if n == 0:
        return {name: 0.0 for name in SIGNAL_FEATURE_SUFFIXES}

    mean_val = float(np.mean(x))
    x_ac = x - mean_val
    std_val = float(np.std(x_ac, ddof=0))
    max_abs = float(np.max(np.abs(x)))
    rms_val = float(np.sqrt(np.mean(x * x)))
    p2p_val = float(np.max(x) - np.min(x))

    if std_val > _EPS:
        z = x_ac / std_val
        skew_val = float(np.mean(z ** 3))
        kurt_val = float(np.mean(z ** 4) - 3.0)
    else:
        skew_val = 0.0
        kurt_val = 0.0

    rms_ac = float(np.sqrt(np.mean(x_ac * x_ac)))
    peak_ac = float(np.max(np.abs(x_ac)))
    crest_val = float(peak_ac / rms_ac) if rms_ac > _EPS else 0.0

    # Single-sided Hann-windowed amplitude spectrum on AC component (immune to DC drift)
    window = np.hanning(n)
    win_sum = float(np.sum(window))
    if win_sum <= _EPS:
        win_sum = float(n)
    fft_vals = np.fft.rfft(x_ac * window)
    amp = (2.0 / win_sum) * np.abs(fft_vals)
    if len(amp) > 0:
        amp[0] = 0.0
    if n % 2 == 0 and len(amp) > 1:
        amp[-1] *= 0.5

    freqs = np.fft.rfftfreq(n, d=1.0 / float(sample_rate_hz))
    df = float(freqs[1] - freqs[0]) if len(freqs) > 1 else float(sample_rate_hz) / max(n, 1)
    nyquist_hz = 0.5 * float(sample_rate_hz)

    amp_sum = float(np.sum(amp))
    spectral_centroid = float(np.sum(freqs * amp) / amp_sum) if amp_sum > _EPS else 0.0
    spectrum_area = float(np.trapezoid(amp, freqs)) if len(freqs) > 1 else float(amp_sum)

    hf_cutoff = min(float(hf_bearing_min_hz), 0.5 * nyquist_hz)
    hf_mask = freqs >= hf_cutoff
    hf_bearing_energy = float(np.sum(amp[hf_mask] ** 2)) if np.any(hf_mask) else 0.0

    f_rot = float(rpm) / 60.0
    h1 = _harmonic_amplitude(freqs, amp, f_rot, nyquist_hz, df)
    h2 = _harmonic_amplitude(freqs, amp, 2.0 * f_rot, nyquist_hz, df)
    h3 = _harmonic_amplitude(freqs, amp, 3.0 * f_rot, nyquist_hz, df)

    return {
        "mean": round(mean_val, 6),
        "std": round(std_val, 6),
        "max": round(max_abs, 6),
        "rms": round(rms_val, 6),
        "peak_to_peak": round(p2p_val, 6),
        "skewness": round(skew_val, 6),
        "kurtosis": round(kurt_val, 6),
        "crest_factor": round(crest_val, 6),
        "spectral_centroid": round(spectral_centroid, 6),
        "spectrum_area": round(spectrum_area, 6),
        "hf_bearing_energy": round(hf_bearing_energy, 6),
        "harmonic_1x": round(h1, 6),
        "harmonic_2x": round(h2, 6),
        "harmonic_3x": round(h3, 6),
    }


def extract_28_features(
    *,
    vibration_g: list[float] | np.ndarray,
    current_a: Optional[list[float] | np.ndarray],
    sample_rate_hz: float,
    rpm: float,
    hf_bearing_min_hz: float = 250.0,
) -> dict[str, float]:
    """Extract all 28 vibration + current features in canonical order."""
    vib_feats = extract_signal_features(
        vibration_g,
        sample_rate_hz=sample_rate_hz,
        rpm=rpm,
        hf_bearing_min_hz=hf_bearing_min_hz,
    )
    if current_a is not None and len(current_a) > 0:
        curr_feats = extract_signal_features(
            current_a,
            sample_rate_hz=sample_rate_hz,
            rpm=rpm,
            hf_bearing_min_hz=hf_bearing_min_hz,
        )
    else:
        curr_feats = {suffix: 0.0 for suffix in SIGNAL_FEATURE_SUFFIXES}

    combined: dict[str, float] = {}
    for suffix in SIGNAL_FEATURE_SUFFIXES:
        combined[f"vib_{suffix}"] = vib_feats[suffix]
    for suffix in SIGNAL_FEATURE_SUFFIXES:
        combined[f"curr_{suffix}"] = curr_feats[suffix]

    # Guarantee exact 28-key ordering
    return {name: combined[name] for name in FEATURE_NAMES}
