"""Integration tests for /api/v1/dashboard/{machine_id}, localization, SSE, CORS, and docs."""

from __future__ import annotations

import json
import threading
import time

from fastapi.testclient import TestClient


def test_dashboard_404_localization_and_compact_payload(client: TestClient) -> None:
    missing = client.get("/api/v1/dashboard/unknown-motor")
    assert missing.status_code == 404

    # Inject a moderate bearing wear fault -> Tier 1
    inj = client.post(
        "/api/v1/inject/",
        json={
            "machine_id": "lathe-hi",
            "fault_type": "bearing_wear",
            "severity": 0.32,
            "temperature_c": 48.0,
            "seed": 11,
        },
    )
    assert inj.status_code == 200

    en_resp = client.get("/api/v1/dashboard/lathe-hi", params={"language": "en"})
    hi_resp = client.get("/api/v1/dashboard/lathe-hi", params={"language": "hi"})
    assert en_resp.status_code == 200
    assert hi_resp.status_code == 200

    # Ensure ultra-lightweight JSON payload (< 512 bytes) for weak 2G/3G networks
    assert len(en_resp.content) < 512
    assert len(hi_resp.content) < 512

    en_data = en_resp.json()
    hi_data = hi_resp.json()
    assert en_data["tier"] == 1
    assert en_data["status"] == "needs_attention"
    assert en_data["language"] == "en"
    assert hi_data["language"] == "hi"
    assert en_data["headline"] != hi_data["headline"]
    assert "बियरिंग" in hi_data["headline"] or "बियरिंग" in hi_data["action_hint"]

    bad_lang = client.get("/api/v1/dashboard/lathe-hi", params={"language": "fr"})
    assert bad_lang.status_code == 422


def test_dashboard_sse_initial_and_live_update(client: TestClient) -> None:
    # Seed initial healthy state
    r0 = client.post(
        "/api/v1/inject/",
        json={
            "machine_id": "sse-motor",
            "fault_type": "healthy",
            "severity": 0.1,
            "seed": 1,
        },
    )
    assert r0.status_code == 200

    # Single-event stream returns initial snapshot immediately
    with client.stream(
        "GET",
        "/api/v1/dashboard/sse-motor/stream",
        params={"language": "en", "max_events": 1},
    ) as stream_resp:
        assert stream_resp.status_code == 200
        assert stream_resp.headers["content-type"].startswith("text/event-stream")
        raw = "".join(stream_resp.iter_text())
    assert "event: status" in raw
    data_line = next(line for line in raw.splitlines() if line.startswith("data: "))
    event_payload = json.loads(data_line.removeprefix("data: "))
    assert event_payload["tier"] == 0
    assert event_payload["status"] == "healthy"

    # Live two-event stream: initial healthy -> pushed critical fault from background thread
    def _trigger_fault() -> None:
        time.sleep(0.15)
        client.post(
            "/api/v1/inject/",
            json={
                "machine_id": "sse-motor",
                "fault_type": "bearing_wear",
                "severity": 0.9,
                "seed": 3,
            },
        )

    worker = threading.Thread(target=_trigger_fault, daemon=True)
    worker.start()

    with client.stream(
        "GET",
        "/api/v1/dashboard/sse-motor/stream",
        params={"language": "hi", "max_events": 2},
    ) as stream_resp2:
        assert stream_resp2.status_code == 200
        raw2 = "".join(stream_resp2.iter_text())

    worker.join(timeout=2.0)
    data_lines = [line for line in raw2.splitlines() if line.startswith("data: ")]
    assert len(data_lines) == 2
    first = json.loads(data_lines[0].removeprefix("data: "))
    second = json.loads(data_lines[1].removeprefix("data: "))
    assert first["tier"] == 0
    assert second["tier"] == 2
    assert second["status"] == "about_to_fail"
    assert second["language"] == "hi"


def test_health_docs_and_cors_headers(client: TestClient) -> None:
    health = client.get("/health", headers={"Origin": "http://localhost:3000"})
    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert health.headers.get("access-control-allow-origin") == "*"

    docs = client.get("/docs")
    assert docs.status_code == 200
    openapi = client.get("/openapi.json")
    assert openapi.status_code == 200
    paths = openapi.json()["paths"]
    assert "/api/v1/recordings/ingest" in paths
    assert "/api/v1/inject/" in paths
    assert "/api/v1/dashboard/{machine_id}" in paths
    assert "/api/v1/dashboard/{machine_id}/stream" in paths
