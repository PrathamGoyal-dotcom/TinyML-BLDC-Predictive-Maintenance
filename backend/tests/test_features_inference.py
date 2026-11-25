"""Unit tests for 28-feature extraction and TinyML fault classification."""

from __future__ import annotations

from app.ml.fault_injector import inject_fault
from app.ml.features import extract_28_features
from app.ml.inference import run_tinyml_inference
from app.schemas import FEATURE_NAMES


def test_extract_28_features_complete_and_without_current() -> None:
    vib, curr, _ = inject_fault(
        fault_type="imbalance",
        severity=0.6,
        snr_db=30.0,
        drift_offset=0.25,
        sample_rate_hz=1000.0,
        rpm=1500.0,
        load_pct=50.0,
        window_size=256,
        seed=42,
    )
    feats = extract_28_features(
        vibration_g=vib,
        current_a=curr,
        sample_rate_hz=1000.0,
        rpm=1500.0,
    )
    assert tuple(feats.keys()) == FEATURE_NAMES
    assert len(feats) == 28
    assert abs(feats["vib_mean"] - 0.25) < 0.08
    assert feats["vib_harmonic_1x"] > feats["vib_harmonic_2x"] * 2.0

    feats_no_curr = extract_28_features(
        vibration_g=vib,
        current_a=None,
        sample_rate_hz=1000.0,
        rpm=1500.0,
    )
    assert len(feats_no_curr) == 28
    assert all(
        feats_no_curr[k] == 0.0 for k in FEATURE_NAMES if k.startswith("curr_")
    )


def test_tinyml_inference_recognizes_all_fault_classes() -> None:
    cases = [
        ("healthy", 0.1, 0),
        ("imbalance", 0.85, 2),
        ("misalignment", 0.85, 2),
        ("bearing_wear", 0.85, 2),
        ("demagnetization", 0.85, 2),
        ("combined", 0.9, 2),
    ]
    for fault_type, severity, expected_tier in cases:
        vib, curr, temp = inject_fault(
            fault_type=fault_type,  # type: ignore[arg-type]
            severity=severity,
            snr_db=25.0,
            drift_offset=0.1,
            sample_rate_hz=1000.0,
            rpm=1500.0,
            load_pct=50.0,
            window_size=256,
            seed=7,
        )
        feats = extract_28_features(
            vibration_g=vib,
            current_a=curr,
            sample_rate_hz=1000.0,
            rpm=1500.0,
        )
        res = run_tinyml_inference(feats, temperature_c=temp, rpm=1500.0)
        assert res.predicted_fault == fault_type, f"Mismatch for {fault_type}: {res}"
        assert res.health_tier == expected_tier, f"Tier mismatch for {fault_type}: {res}"


def test_moderate_fault_maps_to_needs_attention_tier_1() -> None:
    vib, curr, temp = inject_fault(
        fault_type="imbalance",
        severity=0.3,
        snr_db=25.0,
        drift_offset=0.0,
        sample_rate_hz=1000.0,
        rpm=1500.0,
        load_pct=45.0,
        temperature_c=45.0,
        window_size=256,
        seed=12,
    )
    feats = extract_28_features(
        vibration_g=vib,
        current_a=curr,
        sample_rate_hz=1000.0,
        rpm=1500.0,
    )
    res = run_tinyml_inference(feats, temperature_c=temp, rpm=1500.0)
    assert res.health_tier == 1
    assert res.health_status == "needs_attention"
    assert res.predicted_fault == "imbalance"
