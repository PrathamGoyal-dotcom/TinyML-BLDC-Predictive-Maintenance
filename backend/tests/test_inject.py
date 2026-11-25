"""Integration tests for /api/v1/inject/ failure injection pipeline."""

from __future__ import annotations

import math

from fastapi.testclient import TestClient


def test_inject_all_fault_types_and_live_dashboard_transition(client: TestClient) -> None:
    machine_id = "chaff-cutter-01"
    expected_faults = [
        ("healthy", 0.1, 0),
        ("imbalance", 0.85, 2),
        ("misalignment", 0.85, 2),
        ("bearing_wear", 0.85, 2),
        ("demagnetization", 0.85, 2),
        ("combined", 0.9, 2),
    ]

    for fault_type, severity, expected_tier in expected_faults:
        resp = client.post(
            "/api/v1/inject/",
            json={
                "machine_id": machine_id,
                "motor_id": "bldc-24v",
                "fault_type": fault_type,
                "severity": severity,
                "snr_db": 26.0,
                "drift_offset": 0.05,
                "persist": False,
                "seed": 42,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["persisted"] is False
        assert data["inference"]["predicted_fault"] == fault_type
        assert data["inference"]["health_tier"] == expected_tier
        assert len(data["features"]) == 28

        # Live dashboard immediately reflects the injected fault without needing DB persistence
        dash = client.get(f"/api/v1/dashboard/{machine_id}")
        assert dash.status_code == 200
        assert dash.json()["fault"] == fault_type
        assert dash.json()["tier"] == expected_tier

    # persist=False should leave recordings table empty
    recs = client.get("/api/v1/recordings/")
    assert recs.status_code == 200
    assert recs.json()["total"] == 0


def test_inject_with_custom_base_waveform_and_persistence(client: TestClient) -> None:
    base_vib = [0.02 * math.sin(2.0 * math.pi * 25.0 * i / 1000.0) for i in range(256)]
    resp = client.post(
        "/api/v1/inject/",
        json={
            "machine_id": "pump-custom",
            "motor_id": "bldc-48v",
            "fault_type": "misalignment",
            "severity": 0.75,
            "snr_db": 25.0,
            "drift_offset": 0.35,
            "base_vibration_g": base_vib,
            "persist": True,
            "seed": 5,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["persisted"] is True
    assert len(body["vibration_g"]) == 256
    assert abs(body["features"]["vib_mean"] - 0.35) < 0.1
    assert body["inference"]["predicted_fault"] == "misalignment"

    recs = client.get("/api/v1/recordings/", params={"machine_id": "pump-custom"})
    assert recs.status_code == 200
    assert recs.json()["total"] == 1
    assert recs.json()["items"][0]["is_synthetic"] is True
    assert recs.json()["items"][0]["label"] == "misalignment"


def test_inject_severity_bounds_validation(client: TestClient) -> None:
    too_low = client.post(
        "/api/v1/inject/",
        json={"fault_type": "imbalance", "severity": 0.01},
    )
    assert too_low.status_code == 422

    too_high = client.post(
        "/api/v1/inject/",
        json={"fault_type": "imbalance", "severity": 1.5},
    )
    assert too_high.status_code == 422
