"""Lightweight TinyML inference engine mapping 28 features to fault classes & 3 health tiers."""

from __future__ import annotations

import math

from app.core.localization import TIER_TO_STATUS
from app.schemas import FaultType, InferenceResult


def _clip01(val: float) -> float:
    return max(0.0, min(1.0, float(val)))


def _softmax(scores: dict[FaultType, float]) -> dict[FaultType, float]:
    max_s = max(scores.values())
    exps = {k: math.exp(v - max_s) for k, v in scores.items()}
    total = sum(exps.values())
    return {k: v / total for k, v in exps.items()}


def run_tinyml_inference(
    features: dict[str, float],
    *,
    temperature_c: float,
    rpm: float = 1500.0,
) -> InferenceResult:
    """Classify BLDC motor state from the 28 extracted features + thermal context.

    Designed to mirror an edge-deployable quantized decision/scoring head:
    - Immune to DC accelerometer drift (uses AC std/harmonics/kurtosis/HF energy)
    - Separates 1x imbalance, 2x/3x misalignment, impulsive HF bearing wear,
      current/torque-ripple demagnetization, and multi-fault combinations.
    """
    vib_std = features["vib_std"]
    vib_p2p = features["vib_peak_to_peak"]
    vib_kurt = max(0.0, features["vib_kurtosis"])
    vib_crest = features["vib_crest_factor"]
    vib_centroid = features["vib_spectral_centroid"]
    vib_hf = features["vib_hf_bearing_energy"]
    h1 = features["vib_harmonic_1x"]
    h2 = features["vib_harmonic_2x"]
    h3 = features["vib_harmonic_3x"]

    curr_std = features["curr_std"]
    curr_p2p = features["curr_peak_to_peak"]
    curr_area = features["curr_spectrum_area"]
    curr_h1 = features["curr_harmonic_1x"]
    curr_h2 = features["curr_harmonic_2x"]

    # Normalized fault-family activation scores (0 = baseline, >0.28 = active fault)
    # 1. Rotor Imbalance: 1x rotational harmonic dominant over 2x/3x
    h23 = h2 + h3
    imbalance_act = _clip01(
        0.45 * _clip01((h1 - 0.09) / 1.15)
        + 0.55 * _clip01((h1 - 1.15 * h23 - 0.05) / 1.15)
    )

    # 2. Shaft Misalignment: 2x & 3x harmonics strongly elevated
    misalign_raw = (
        max(0.0, (h2 - 0.06) / 0.70)
        + max(0.0, (h3 - 0.04) / 0.55)
    ) * 0.65
    if h23 > 1.2 * max(h1, 0.05):
        misalign_raw += 0.15 * _clip01((h23 - 0.12) / 0.6)
    misalign_act = _clip01(misalign_raw)

    # 3. Bearing Wear: high-frequency burst energy + kurtosis + crest factor + high centroid
    bearing_raw = (
        0.55 * _clip01((vib_hf - 0.005) / 0.12)
        + 0.25 * _clip01((vib_kurt - 0.6) / 5.5)
        + 0.25 * _clip01((vib_crest - 2.6) / 2.6)
        + 0.15 * _clip01((vib_centroid - 110.0) / 180.0)
    )
    bearing_act = _clip01(bearing_raw)

    # 4. Demagnetization / Torque Ripple: strong AC current ripple & low/mid current harmonics
    demag_raw = (
        0.45 * _clip01((curr_std - 0.14) / 1.15)
        + 0.30 * _clip01((curr_h1 + curr_h2 - 0.18) / 1.40)
        + 0.25 * _clip01((curr_p2p - 0.50) / 3.50)
        + 0.15 * _clip01((curr_area - 4.0) / 30.0)
    )
    demag_act = _clip01(demag_raw)

    # Count how many independent fault mechanisms are simultaneously active
    active_families = sum(
        act >= 0.22
        for act in (imbalance_act, misalign_act, bearing_act, demag_act)
    )
    combined_act = 0.0
    if active_families >= 2:
        top_two = sorted(
            [imbalance_act, misalign_act, bearing_act, demag_act], reverse=True
        )[:2]
        combined_act = _clip01(0.55 * (top_two[0] + top_two[1]) + 0.12 * (active_families - 2))

    # Global mechanical + electrical + thermal anomaly score
    mech_energy = _clip01((vib_std - 0.055) / 0.90)
    peak_stress = _clip01((vib_p2p - 0.25) / 3.40)
    thermal_stress = _clip01((float(temperature_c) - 62.0) / 35.0)

    max_fault_act = max(
        imbalance_act, misalign_act, bearing_act, demag_act, combined_act
    )
    blended_score = (
        0.65 * max_fault_act
        + 0.15 * mech_energy
        + 0.12 * peak_stress
        + 0.08 * thermal_stress
    )
    if max_fault_act >= 0.80:
        blended_score = max(blended_score, 0.72 * max_fault_act)
    anomaly_score = _clip01(blended_score)

    # If motor is idle (rpm == 0) and vibration is negligible, keep healthy unless overheated
    if rpm <= 1.0 and vib_std < 0.05 and temperature_c < 70.0:
        anomaly_score = min(anomaly_score, 0.12)

    # Logits for compact softmax confidence across all 6 classes
    healthy_logit = 2.8 - 6.0 * max_fault_act - 2.0 * thermal_stress
    logits: dict[FaultType, float] = {
        "healthy": healthy_logit,
        "imbalance": -0.5 + 5.2 * imbalance_act - 0.8 * (combined_act > 0.35),
        "misalignment": -0.5 + 5.2 * misalign_act - 0.8 * (combined_act > 0.35),
        "bearing_wear": -0.5 + 5.4 * bearing_act - 0.8 * (combined_act > 0.35),
        "demagnetization": -0.5 + 5.2 * demag_act - 0.8 * (combined_act > 0.35),
        "combined": -1.2 + 5.8 * combined_act if active_families >= 2 else -5.0,
    }
    probs = _softmax(logits)
    predicted_fault: FaultType = max(probs, key=lambda k: probs[k])
    confidence = round(float(probs[predicted_fault]), 3)

    if anomaly_score < 0.30:
        tier = 0
        if predicted_fault != "healthy" and anomaly_score < 0.20:
            predicted_fault = "healthy"
            confidence = round(max(probs["healthy"], 1.0 - anomaly_score), 3)
    elif anomaly_score < 0.62:
        tier = 1
    else:
        tier = 2

    return InferenceResult(
        health_tier=tier,
        health_status=TIER_TO_STATUS[tier],
        predicted_fault=predicted_fault,
        confidence=_clip01(confidence),
        anomaly_score=round(anomaly_score, 3),
    )
