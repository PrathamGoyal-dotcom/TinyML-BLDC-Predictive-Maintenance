#!/usr/bin/env python3
"""End-to-end <2-minute reviewer demo for the TinyML BLDC Predictive Maintenance backend."""

from __future__ import annotations

import argparse
import csv
import io
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Optional

import httpx

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _ensure_utf8_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def _run_cmd(cmd: list[str], cwd: Path) -> None:
    print(f"$ {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(cwd), check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {proc.returncode}: {' '.join(cmd)}")


def _wait_for_ready(client: httpx.Client, base_url: str, timeout_s: float = 60.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    last_err: Optional[Exception] = None
    while time.monotonic() < deadline:
        try:
            health = client.get(f"{base_url}/health")
            docs = client.get(f"{base_url}/docs")
            if health.status_code == 200 and docs.status_code == 200:
                return health.json()
        except httpx.HTTPError as exc:
            last_err = exc
        time.sleep(1.0)
    raise TimeoutError(f"Backend did not become ready within {timeout_s}s ({last_err})")


def run_demo(
    *,
    base_url: str = "http://localhost:8000",
    machine_id: str = "pump-bldc-01",
    skip_docker: bool = False,
    teardown: bool = False,
    export_path: Path = _REPO_ROOT / "data" / "demo_export.csv",
) -> dict[str, Any]:
    """Execute the 4-step end-to-end reviewer demo and return summary metrics."""
    _ensure_utf8_stdout()
    started = time.perf_counter()
    summary: dict[str, Any] = {"machine_id": machine_id, "base_url": base_url}

    try:
        if not skip_docker:
            print("\n[1/4] Starting backend via docker compose up --build -d ...")
            _run_cmd(["docker", "compose", "up", "--build", "-d"], cwd=_REPO_ROOT)
        else:
            print(f"\n[1/4] Checking running backend at {base_url} (--skip-docker) ...")

        with httpx.Client(timeout=15.0) as client:
            health_json = _wait_for_ready(client, base_url, timeout_s=60.0)
            summary["health"] = health_json
            print(f"      /health -> {health_json} | /docs -> 200 OK")

            print(f"\n[2/4] Streaming 5 healthy BLDC windows for '{machine_id}' via serial_bridge.py --simulate ...")
            _run_cmd(
                [
                    sys.executable,
                    str(_REPO_ROOT / "scripts" / "serial_bridge.py"),
                    "--simulate",
                    "--fault",
                    "healthy",
                    "--severity",
                    "0.1",
                    "--count",
                    "5",
                    "--interval",
                    "0.2",
                    "--machine-id",
                    machine_id,
                    "--url",
                    f"{base_url}/api/v1/recordings/ingest",
                ],
                cwd=_REPO_ROOT,
            )

            dash_healthy = client.get(
                f"{base_url}/api/v1/dashboard/{machine_id}",
                params={"language": "hi"},
            )
            dash_healthy.raise_for_status()
            d_healthy = dash_healthy.json()
            assert d_healthy["status"] == "healthy", f"Expected healthy, got {d_healthy}"
            summary["healthy_dashboard_hi"] = d_healthy
            print(
                f"      Dashboard (hi): tier={d_healthy['tier']} status={d_healthy['status']} "
                f"| {d_healthy['headline']} — {d_healthy['action_hint']}"
            )

            print("\n[3/4] Injecting controlled faults (imbalance 0.45 -> bearing_wear 0.95) ...")
            inj_imb = client.post(
                f"{base_url}/api/v1/inject/",
                json={
                    "machine_id": machine_id,
                    "fault_type": "imbalance",
                    "severity": 0.45,
                    "persist": True,
                    "seed": 42,
                },
            )
            inj_imb.raise_for_status()
            dash_imb = client.get(
                f"{base_url}/api/v1/dashboard/{machine_id}",
                params={"language": "hi"},
            )
            dash_imb.raise_for_status()
            d_imb = dash_imb.json()
            assert d_imb["status"] == "needs_attention", f"Expected needs_attention, got {d_imb}"
            summary["imbalance_dashboard_hi"] = d_imb
            print(
                f"      After imbalance (0.45): tier={d_imb['tier']} status={d_imb['status']} "
                f"fault={d_imb['fault']} score={d_imb['anomaly_score']} | {d_imb['headline']}"
            )

            inj_bear = client.post(
                f"{base_url}/api/v1/inject/",
                json={
                    "machine_id": machine_id,
                    "fault_type": "bearing_wear",
                    "severity": 0.95,
                    "persist": True,
                    "seed": 42,
                },
            )
            inj_bear.raise_for_status()
            dash_bear = client.get(
                f"{base_url}/api/v1/dashboard/{machine_id}",
                params={"language": "hi"},
            )
            dash_bear.raise_for_status()
            d_bear = dash_bear.json()
            assert d_bear["status"] == "about_to_fail", f"Expected about_to_fail, got {d_bear}"
            summary["bearing_wear_dashboard_hi"] = d_bear
            print(
                f"      After bearing_wear (0.95): tier={d_bear['tier']} status={d_bear['status']} "
                f"fault={d_bear['fault']} score={d_bear['anomaly_score']} | {d_bear['headline']}"
            )

            print("\n[4/4] Running DUDU-BLDC baseline evaluation & exporting training CSV ...")
            eval_resp = client.post(
                f"{base_url}/api/v1/baseline/evaluate",
                json={
                    "source": "synthetic",
                    "n_recordings": 60,
                    "windows_per_recording": 3,
                    "n_splits": 5,
                    "model_type": "mlp",
                    "seed": 42,
                },
            )
            eval_resp.raise_for_status()
            eval_json = eval_resp.json()
            summary["baseline"] = eval_json

            csv_resp = client.get(
                f"{base_url}/api/v1/recordings/export/csv",
                params={"machine_id": machine_id},
            )
            csv_resp.raise_for_status()
            export_path.parent.mkdir(parents=True, exist_ok=True)
            export_path.write_text(csv_resp.text, encoding="utf-8")
            rows = list(csv.DictReader(io.StringIO(csv_resp.text)))
            summary["exported_csv_path"] = str(export_path)
            summary["exported_rows"] = len(rows)

            elapsed_s = round(time.perf_counter() - started, 2)
            summary["elapsed_seconds"] = elapsed_s

            print("\n==================== DEMO SUMMARY ====================")
            print(f"Elapsed Time          : {elapsed_s:.2f} s")
            print(f"Health & OpenAPI      : /health=ok, /docs=200")
            print(
                f"Healthy (hi)          : tier={d_healthy['tier']} ({d_healthy['status']}) | {d_healthy['headline']}"
            )
            print(
                f"Imbalance (0.45, hi)  : tier={d_imb['tier']} ({d_imb['status']}) | fault={d_imb['fault']} | {d_imb['headline']}"
            )
            print(
                f"Bearing (0.95, hi)    : tier={d_bear['tier']} ({d_bear['status']}) | fault={d_bear['fault']} | {d_bear['headline']}"
            )
            print(
                f"Baseline (MLP, 5-fold): acc={eval_json['overall_accuracy']*100:.2f}% "
                f"bal_acc={eval_json['balanced_accuracy']*100:.2f}% "
                f"int8={eval_json['quantized_int8_accuracy']*100:.2f}% "
                f"latency={eval_json['avg_inference_latency_ms']:.4f}ms "
                f"footprint={eval_json['model_footprint_kb']['float32_kb']}KB/"
                f"{eval_json['model_footprint_kb']['int8_quantized_kb']}KB"
            )
            print(f"Exported Dataset CSV  : {len(rows)} rows -> {export_path}")
            print("======================================================")
            return summary
    finally:
        if teardown and not skip_docker:
            print("\nStopping Docker Compose stack (--teardown)...")
            _run_cmd(["docker", "compose", "down"], cwd=_REPO_ROOT)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the <2-minute end-to-end reviewer demo against the Dockerized backend."
    )
    parser.add_argument("--base-url", default="http://localhost:8000", help="Backend base URL")
    parser.add_argument("--machine-id", default="pump-bldc-01", help="Target machine_id")
    parser.add_argument(
        "--skip-docker",
        action="store_true",
        help="Skip `docker compose up --build -d` and hit an already-running server",
    )
    parser.add_argument(
        "--teardown",
        action="store_true",
        help="Run `docker compose down` after completing the demo",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    run_demo(
        base_url=args.base_url,
        machine_id=args.machine_id,
        skip_docker=args.skip_docker,
        teardown=args.teardown,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
