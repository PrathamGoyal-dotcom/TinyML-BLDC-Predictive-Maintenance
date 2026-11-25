"""Integration tests for /api/v1/recordings/* endpoints."""

from __future__ import annotations

import csv
import io

from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from app.ml.fault_injector import inject_fault
from app.schemas import FEATURE_NAMES


def _sample_payload(
    *,
    recording_id: str = "rec-001",
    machine_id: str = "pump-01",
    fault_type: str = "healthy",
    severity: float = 0.1,
    include_current: bool = True,
    label: str | None = "healthy",
) -> dict:
    vib, curr, temp = inject_fault(
        fault_type=fault_type,  # type: ignore[arg-type]
        severity=severity,
        snr_db=28.0,
        drift_offset=0.0,
        sample_rate_hz=1000.0,
        rpm=1500.0,
        load_pct=50.0,
        window_size=256,
        seed=99,
    )
    payload = {
        "machine_id": machine_id,
        "recording_id": recording_id,
        "motor_id": "bldc-m1",
        "sample_rate_hz": 1000.0,
        "rpm": 1500.0,
        "load_pct": 50.0,
        "label": label,
        "vibration_g": vib,
        "temperature_c": temp,
    }
    if include_current:
        payload["current_a"] = curr
    return payload


def test_ingest_list_and_csv_export(client: TestClient) -> None:
    r1 = client.post(
        "/api/v1/recordings/ingest",
        json=_sample_payload(recording_id="rec-1", machine_id="pump-01", fault_type="healthy"),
    )
    assert r1.status_code == 201
    body1 = r1.json()
    assert len(body1["features"]) == 28
    assert body1["inference"]["health_tier"] == 0
    assert body1["dashboard"]["status"] == "healthy"

    # Ingest without current_a and with bearing wear fault
    r2 = client.post(
        "/api/v1/recordings/ingest",
        json=_sample_payload(
            recording_id="rec-2",
            machine_id="lathe-02",
            fault_type="bearing_wear",
            severity=0.85,
            include_current=False,
            label="bearing_wear",
        ),
    )
    assert r2.status_code == 201
    body2 = r2.json()
    assert body2["features"]["curr_rms"] == 0.0
    assert body2["inference"]["predicted_fault"] == "bearing_wear"
    assert body2["inference"]["health_tier"] == 2

    # Duplicate recording_id -> 409 Conflict
    dup = client.post(
        "/api/v1/recordings/ingest",
        json=_sample_payload(recording_id="rec-1"),
    )
    assert dup.status_code == 409

    # List recordings with filters and include_raw=false
    listed = client.get("/api/v1/recordings/", params={"machine_id": "lathe-02", "include_raw": "false"})
    assert listed.status_code == 200
    list_data = listed.json()
    assert list_data["total"] == 1
    assert list_data["items"][0]["recording_id"] == "rec-2"
    assert list_data["items"][0]["vibration_g"] is None

    # CSV export includes all 28 features and both rows
    csv_resp = client.get("/api/v1/recordings/export/csv")
    assert csv_resp.status_code == 200
    assert "text/csv" in csv_resp.headers["content-type"]
    rows = list(csv.DictReader(io.StringIO(csv_resp.text)))
    assert len(rows) == 2
    for feat in FEATURE_NAMES:
        assert feat in rows[0]
    assert rows[0]["recording_id"] == "rec-1"
    assert rows[1]["recording_id"] == "rec-2"


def test_ingest_validation_errors(client: TestClient) -> None:
    bad = _sample_payload(recording_id="rec-bad")
    bad["vibration_g"] = [0.1, 0.2, 0.3]  # fewer than 8 samples
    resp = client.post("/api/v1/recordings/ingest", json=bad)
    assert resp.status_code == 422


def test_dashboard_state_hydrates_from_sqlite_on_restart(test_settings: Settings) -> None:
    app1 = create_app(test_settings)
    with TestClient(app1) as c1:
        r = c1.post(
            "/api/v1/recordings/ingest",
            json=_sample_payload(recording_id="rec-persist", machine_id="mill-09"),
        )
        assert r.status_code == 201

    app2 = create_app(test_settings)
    with TestClient(app2) as c2:
        dash = c2.get("/api/v1/dashboard/mill-09")
        assert dash.status_code == 200
        assert dash.json()["machine_id"] == "mill-09"
        assert dash.json()["tier"] == 0
