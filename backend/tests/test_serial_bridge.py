"""Tests for scripts/serial_bridge.py NDJSON parsing and --simulate mode."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from scripts.serial_bridge import build_parser, parse_serial_line, run_bridge


def test_parse_serial_line_handles_valid_empty_and_bad_json() -> None:
    assert parse_serial_line("   \n") is None

    line = json.dumps(
        {
            "sample_rate_hz": 1000.0,
            "rpm": 1500.0,
            "load_pct": 40.0,
            "vibration_g": [0.01] * 16,
            "temperature_c": 39.5,
        }
    )
    parsed = parse_serial_line(line, default_machine_id="esp32-01", default_motor_id="m-01")
    assert parsed is not None
    assert parsed["machine_id"] == "esp32-01"
    assert parsed["motor_id"] == "m-01"
    assert parsed["recording_id"].startswith("ser-")

    with pytest.raises(json.JSONDecodeError):
        parse_serial_line("{broken-json")


def test_serial_bridge_simulate_posts_to_ingest(client: TestClient) -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--simulate",
            "--fault",
            "cycle",
            "--count",
            "3",
            "--interval",
            "0",
            "--machine-id",
            "sim-rig-01",
            "--url",
            "http://testserver/api/v1/recordings/ingest",
        ]
    )

    # TestClient is an httpx.Client wired directly to the FastAPI ASGI app
    sent = run_bridge(args, http_client=client)

    assert sent == 3
    recs = client.get("/api/v1/recordings/", params={"machine_id": "sim-rig-01"})
    assert recs.status_code == 200
    assert recs.json()["total"] == 3

    dash = client.get("/api/v1/dashboard/sim-rig-01")
    assert dash.status_code == 200
    assert dash.json()["machine_id"] == "sim-rig-01"
