"""Synthesize healthy BLDC waveforms and inject controlled physical fault signatures."""

from __future__ import annotations

from typing import Optional

import numpy as np

from app.schemas import FaultType


def _add_awgn(
    signal: np.ndarray, snr_db: float, rng: np.random.Generator
) -> np.ndarray:
    """Add zero-mean Gaussian noise to achieve target SNR (dB) on the AC component."""
    ac = signal - np.mean(signal)
    rms_ac = float(np.sqrt(np.mean(ac * ac)))
    base_rms = max(rms_ac, 0.02)
    noise_std = base_rms * (10.0 ** (-float(snr_db) / 20.0))
    return signal + rng.normal(0.0, noise_std, size=len(signal))


def synthesize_healthy_bldc(
    *,
    window_size: int,
    sample_rate_hz: float,
    rpm: float,
    load_pct: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate realistic healthy BLDC vibration (g) and stator current (A) windows."""
    t = np.arange(window_size, dtype=np.float64) / float(sample_rate_hz)
    f0 = float(rpm) / 60.0
    load_factor = 0.6 + 0.4 * (float(load_pct) / 100.0)

    # Low baseline 1x/2x/3x mechanical harmonics + faint commutation ripple
    vib = (
        0.032 * load_factor * np.sin(2.0 * np.pi * f0 * t + 0.2)
        + 0.015 * load_factor * np.sin(2.0 * np.pi * 2.0 * f0 * t + 0.7)
        + 0.008 * load_factor * np.sin(2.0 * np.pi * 3.0 * f0 * t + 1.1)
    )
    vib += rng.normal(0.0, 0.008, size=window_size)

    # Nominal BLDC phase/bus current: steady DC proportional to load + tiny 6x ripple
    i_dc = 1.2 + 4.0 * (float(load_pct) / 100.0)
    curr = (
        i_dc
        + 0.045 * load_factor * np.sin(2.0 * np.pi * f0 * t + 0.4)
        + 0.035 * load_factor * np.sin(2.0 * np.pi * 6.0 * f0 * t)
    )
    curr += rng.normal(0.0, 0.012, size=window_size)
    return vib, curr


def _bearing_impulse_train(
    t: np.ndarray,
    *,
    f0: float,
    sample_rate_hz: float,
    severity: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate impulsive high-frequency bearing raceway defect bursts."""
    nyquist = 0.5 * float(sample_rate_hz)
    bpfo = max(3.6 * f0, 40.0)
    if bpfo >= 0.45 * nyquist:
        bpfo = 0.2 * nyquist
    resonance_hz = min(max(0.70 * nyquist, 260.0), 0.90 * nyquist)
    period = 1.0 / max(bpfo, 1.0)
    tau = max(0.0025, 0.16 * period)

    out = np.zeros_like(t)
    duration = float(t[-1] - t[0]) if len(t) > 1 else 0.0
    n_impulses = max(1, int(duration / period) + 1)
    amp = 0.35 + 1.55 * severity

    for k in range(n_impulses):
        t0 = k * period + rng.uniform(-0.04 * period, 0.04 * period)
        dt = t - t0
        mask = dt >= 0.0
        if np.any(mask):
            burst = (
                amp
                * np.exp(-dt[mask] / tau)
                * np.sin(2.0 * np.pi * resonance_hz * dt[mask])
            )
            out[mask] += burst
    return out


def inject_fault(
    *,
    fault_type: FaultType,
    severity: float,
    snr_db: float,
    drift_offset: float,
    sample_rate_hz: float,
    rpm: float,
    load_pct: float,
    temperature_c: Optional[float] = None,
    window_size: int = 256,
    base_vibration_g: Optional[list[float]] = None,
    base_current_a: Optional[list[float]] = None,
    seed: Optional[int] = None,
) -> tuple[list[float], list[float], float]:
    """Inject a controlled BLDC fault onto a baseline or synthesized waveform."""
    rng = np.random.default_rng(seed)
    f0 = float(rpm) / 60.0

    if base_vibration_g is not None:
        vib = np.asarray(base_vibration_g, dtype=np.float64).copy()
        n = len(vib)
    else:
        n = window_size
        vib, synth_curr = synthesize_healthy_bldc(
            window_size=n,
            sample_rate_hz=sample_rate_hz,
            rpm=rpm,
            load_pct=load_pct,
            rng=rng,
        )

    if base_current_a is not None:
        curr = np.asarray(base_current_a, dtype=np.float64).copy()
        if len(curr) != n:
            # Resample/pad or match length cleanly if user passed mismatched base lengths
            curr = np.resize(curr, n)
    elif base_vibration_g is not None:
        _, curr = synthesize_healthy_bldc(
            window_size=n,
            sample_rate_hz=sample_rate_hz,
            rpm=rpm,
            load_pct=load_pct,
            rng=rng,
        )
    else:
        curr = synth_curr

    t = np.arange(n, dtype=np.float64) / float(sample_rate_hz)

    def _apply_imbalance(scale: float) -> None:
        nonlocal vib, curr
        s = severity * scale
        vib += (0.22 + 1.15 * s) * np.sin(2.0 * np.pi * f0 * t + 0.35)
        curr += (0.08 + 0.25 * s) * np.sin(2.0 * np.pi * f0 * t + 0.35)

    def _apply_misalignment(scale: float) -> None:
        nonlocal vib, curr
        s = severity * scale
        vib += (0.18 + 0.95 * s) * np.sin(2.0 * np.pi * 2.0 * f0 * t + 0.5)
        vib += (0.14 + 0.72 * s) * np.sin(2.0 * np.pi * 3.0 * f0 * t + 1.2)
        curr += (0.06 + 0.20 * s) * np.sin(2.0 * np.pi * 2.0 * f0 * t)
        curr += (0.05 + 0.16 * s) * np.sin(2.0 * np.pi * 3.0 * f0 * t)

    def _apply_bearing_wear(scale: float) -> None:
        nonlocal vib, curr
        s = severity * scale
        vib += _bearing_impulse_train(
            t,
            f0=f0,
            sample_rate_hz=sample_rate_hz,
            severity=s,
            rng=rng,
        )
        curr += 0.06 * s * np.sin(2.0 * np.pi * 4.0 * f0 * t)

    def _apply_demagnetization(scale: float) -> None:
        nonlocal vib, curr
        s = severity * scale
        # Strong current/torque ripple & stator harmonic distortion
        curr += (0.35 + 1.45 * s) * np.sin(2.0 * np.pi * 0.5 * f0 * t + 0.2)
        curr += (0.28 + 1.15 * s) * np.sin(2.0 * np.pi * f0 * t + 0.9)
        curr += (0.22 + 0.85 * s) * np.sin(2.0 * np.pi * 2.0 * f0 * t + 1.4)
        curr += (0.15 + 0.65 * s) * np.sin(2.0 * np.pi * 4.5 * f0 * t)
        # Torque pulsation causes moderate broad vibration ripple
        vib += (0.08 + 0.32 * s) * np.sin(2.0 * np.pi * 0.5 * f0 * t)
        vib += (0.07 + 0.25 * s) * np.sin(2.0 * np.pi * 1.5 * f0 * t)

    if fault_type == "imbalance":
        _apply_imbalance(1.0)
    elif fault_type == "misalignment":
        _apply_misalignment(1.0)
    elif fault_type == "bearing_wear":
        _apply_bearing_wear(1.0)
    elif fault_type == "demagnetization":
        _apply_demagnetization(1.0)
    elif fault_type == "combined":
        _apply_imbalance(0.75)
        _apply_misalignment(0.75)
        _apply_bearing_wear(0.75)
        _apply_demagnetization(0.75)

    # Apply AWGN at target SNR and sensor DC drift offset
    vib = _add_awgn(vib, snr_db=snr_db, rng=rng) + float(drift_offset)
    curr = _add_awgn(curr, snr_db=max(snr_db, 15.0), rng=rng)

    if temperature_c is None:
        temp_delta = 0.0 if fault_type == "healthy" else (6.0 + 26.0 * severity)
        if fault_type == "demagnetization":
            temp_delta += 8.0 * severity
        elif fault_type == "combined":
            temp_delta += 10.0 * severity
        computed_temp = round(34.0 + 0.16 * float(load_pct) + temp_delta, 2)
    else:
        computed_temp = round(float(temperature_c), 2)

    vib_list = [round(float(v), 5) for v in vib]
    curr_list = [round(float(c), 5) for c in curr]
    return vib_list, curr_list, computed_temp
