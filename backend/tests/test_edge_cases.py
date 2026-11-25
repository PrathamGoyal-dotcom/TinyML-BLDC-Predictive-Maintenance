"""Comprehensive edge-case tests across inference, recordings, SSE, baseline, and serial bridge."""

from __future__ import annotations

import csv
import io
import json
import sys
import threading
import time
from types import SimpleNamespace

from fastapi.testclient import TestClient

from app.ml.fault_injector import inject_fault
from app.ml.features import extract_28_features
from app.ml.inference import run_tinyml_inference
from scripts.serial_bridge import build_parser, run_bridge


def test_idle_motor_zero_rpm_and_empty_current(client: TestClient) -> None:
    resp = client.post(
        "/api/v1/recordings/ingest",
        json={
            "machine_id": "idle-motor-01",
            "recording_id": "rec-idle-01",
            "motor_id": "m-idle",
            "sample_rate_hz": 1000.0,
            "rpm": 0.0,
            "load_pct": 0.0,
            "vibration_g": [0.002, -0.001, 0.003, -0.002] * 16,
            "current_a": [],
            "temperature_c": 32.0,
        },
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["features"]["vib_harmonic_1x"] == 0.0
    assert data["features"]["curr_rms"] == 0.0
    assert data["inference"]["health_tier"] == 0
    assert data["inference"]["predicted_fault"] == "healthy"


def test_recordings_filters_and_exclude_synthetic_csv(client: TestClient) -> None:
    vib, curr, temp = inject_fault(
        fault_type="healthy",
        severity=0.1,
        snr_db=28.0,
        drift_offset=0.0,
        sample_rate_hz=1000.0,
        rpm=1500.0,
        load_pct=40.0,
        window_size=128,
        seed=1,
    )
    r_real = client.post(
        "/api/v1/recordings/ingest",
        json={
            "machine_id": "m-filter",
            "recording_id": "real-1",
            "motor_id": "mot-a",
            "sample_rate_hz": 1000.0,
            "rpm": 1500.0,
            "load_pct": 40.0,
            "label": "healthy",
            "vibration_g": vib,
            "current_a": curr,
            "temperature_c": temp,
        },
    )
    assert r_real.status_code == 201

    r_syn = client.post(
        "/api/v1/inject/",
        json={
            "machine_id": "m-filter",
            "motor_id": "mot-b",
            "recording_id": "syn-1",
            "fault_type": "imbalance",
            "severity": 0.45,
            "persist": True,
            "seed": 42,
        },
    )
    assert r_syn.status_code == 200
    assert r_syn.json()["inference"]["health_tier"] == 1

    by_tier = client.get("/api/v1/recordings/", params={"health_tier": 1, "predicted_fault": "imbalance"})
    assert by_tier.status_code == 200
    assert by_tier.json()["total"] == 1
    assert by_tier.json()["items"][0]["recording_id"] == "syn-1"

    csv_real_only = client.get(
        "/api/v1/recordings/export/csv",
        params={"include_synthetic": "false"},
    )
    assert csv_real_only.status_code == 200
    rows = list(csv.DictReader(io.StringIO(csv_real_only.text)))
    assert len(rows) == 1
    assert rows[0]["recording_id"] == "real-1"


def test_sse_connect_before_first_telemetry_and_heartbeat(client: TestClient) -> None:
    machine_id = "fresh-sse-machine"

    def _send_ingest() -> None:
        time.sleep(0.35)
        vib, curr, temp = inject_fault(
            fault_type="bearing_wear",
            severity=0.95,
            snr_db=25.0,
            drift_offset=0.0,
            sample_rate_hz=1000.0,
            rpm=1500.0,
            load_pct=60.0,
            window_size=256,
            seed=42,
        )
        client.post(
            "/api/v1/recordings/ingest",
            json={
                "machine_id": machine_id,
                "recording_id": "sse-rec-1",
                "motor_id": "mot-sse",
                "sample_rate_hz": 1000.0,
                "rpm": 1500.0,
                "load_pct": 60.0,
                "vibration_g": vib,
                "current_a": curr,
                "temperature_c": temp,
            },
        )

    t = threading.Thread(target=_send_ingest, daemon=True)
    t.start()

    with client.stream(
        "GET",
        f"/api/v1/dashboard/{machine_id}/stream",
        params={"language": "en", "max_events": 1, "heartbeat_s": 0.15},
    ) as resp:
        assert resp.status_code == 200
        raw = "".join(resp.iter_text())

    t.join(timeout=2.0)
    assert f": connected {machine_id}" in raw
    assert ": ping" in raw
    data_line = next(line for line in raw.splitlines() if line.startswith("data: "))
    evt = json.loads(data_line.removeprefix("data: "))
    assert evt["tier"] == 2
    assert evt["status"] == "about_to_fail"
    assert evt["fault"] == "bearing_wear"


def test_baseline_error_paths_and_multiwindow_csv_grouping(
    client: TestClient, tmp_path
) -> None:
    no_path = client.post("/api/v1/baseline/evaluate", json={"source": "csv"})
    assert no_path.status_code == 400

    missing_csv = client.post(
        "/api/v1/baseline/evaluate",
        json={"source": "csv", "csv_path": str(tmp_path / "nope.csv")},
    )
    assert missing_csv.status_code == 404

    empty_db = client.post("/api/v1/baseline/evaluate", json={"source": "database"})
    assert empty_db.status_code == 400

    # Create a multi-window CSV where multiple windows share the same recording_id
    csv_path = tmp_path / "multiwindow.csv"
    header = [
        "recording_id",
        "label",
        *extract_28_features(
            vibration_g=[0.0] * 16,
            current_a=None,
            sample_rate_hz=1000.0,
            rpm=1500.0,
        ).keys(),
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        classes = ["healthy", "imbalance", "misalignment"]
        for rec_idx in range(9):
            lbl = classes[rec_idx % len(classes)]
            for win_idx in range(3):
                vib, curr, _ = inject_fault(
                    fault_type=lbl,  # type: ignore[arg-type]
                    severity=0.15 if lbl == "healthy" else 0.75,
                    snr_db=24.0,
                    drift_offset=0.0,
                    sample_rate_hz=1000.0,
                    rpm=1500.0,
                    load_pct=50.0,
                    window_size=256,
                    seed=rec_idx * 10 + win_idx,
                )
                feats = extract_28_features(
                    vibration_g=vib,
                    current_a=curr,
                    sample_rate_hz=1000.0,
                    rpm=1500.0,
                )
                writer.writerow([f"grp-{rec_idx:02d}", lbl, *feats.values()])

    grouped = client.post(
        "/api/v1/baseline/evaluate",
        json={"source": "csv", "csv_path": str(csv_path), "n_splits": 3},
    )
    assert grouped.status_code == 200
    g_data = grouped.json()
    assert g_data["n_samples"] == 27
    assert g_data["n_recordings"] == 9
    assert g_data["leakage_verified"] is True
    assert g_data["overall_accuracy"] >= 0.85


def test_serial_bridge_hardware_mode_skips_bad_lines(client: TestClient, monkeypatch) -> None:
    vib, curr, temp = inject_fault(
        fault_type="healthy",
        severity=0.1,
        snr_db=26.0,
        drift_offset=0.0,
        sample_rate_hz=1000.0,
        rpm=1500.0,
        load_pct=45.0,
        window_size=64,
        seed=5,
    )
    valid_line = (
        json.dumps(
            {
                "sample_rate_hz": 1000.0,
                "rpm": 1500.0,
                "load_pct": 45.0,
                "vibration_g": vib,
                "current_a": curr,
                "temperature_c": temp,
            }
        )
        + "\n"
    ).encode("utf-8")

    lines = iter([b"\n", b"{corrupt-json\n", valid_line])

    class _FakeSerial:
        def __init__(self, port: str, baud: int, timeout: float = 1.0) -> None:
            self.port = port
            self.baud = baud

        def __enter__(self):
            return self

        def __exit__(self, *args) -> None:
            return None

        def readline(self) -> bytes:
            return next(lines, b"")

    monkeypatch.setitem(sys.modules, "serial", SimpleNamespace(Serial=_FakeSerial))

    args = build_parser().parse_args(
        [
            "--port",
            "COM9",
            "--count",
            "1",
            "--machine-id",
            "esp32-hw-01",
            "--url",
            "http://testserver/api/v1/recordings/ingest",
        ]
    )
    sent = run_bridge(args, http_client=client)
    assert sent == 1
    dash = client.get("/api/v1/dashboard/esp32-hw-01")
    assert dash.status_code == 200
    assert dash.json()["status"] == "healthy"


def test_severity_tier_progression_imbalance_and_bearing() -> None:
    # Verify exact progression: healthy -> needs_attention (0.45) -> about_to_fail (0.95)
    for fault, sev, expected_tier in (
        ("healthy", 0.10, 0),
        ("imbalance", 0.45, 1),
        ("bearing_wear", 0.95, 2),
    ):
        vib, curr, temp = inject_fault(
            fault_type=fault,  # type: ignore[arg-type]
            severity=sev,
            snr_db=25.0,
            drift_offset=0.0,
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
        res = run_tinyml_inference(feats, temperature_c=temp, rpm=1500.0)
        assert res.predicted_fault == fault
        assert res.health_tier == expected_tier
