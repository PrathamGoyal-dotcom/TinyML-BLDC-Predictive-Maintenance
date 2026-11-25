#!/usr/bin/env python3
"""USB Serial NDJSON -> FastAPI /ingest bridge CLI with --simulate mode."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

import httpx

# Allow importing app modules when invoked as `python scripts/serial_bridge.py`
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.ml.fault_injector import inject_fault  # noqa: E402
from app.schemas import FaultType  # noqa: E402

_CYCLE_SEQUENCE: tuple[tuple[FaultType, float], ...] = (
    ("healthy", 0.1),
    ("imbalance", 0.35),
    ("misalignment", 0.55),
    ("bearing_wear", 0.8),
    ("demagnetization", 0.7),
    ("combined", 0.9),
)


def parse_serial_line(
    raw_line: str,
    *,
    default_machine_id: str = "bldc-01",
    default_motor_id: str = "motor-01",
) -> Optional[dict[str, Any]]:
    """Parse a single newline-delimited JSON line from ESP32/Arduino telemetry."""
    text = raw_line.strip()
    if not text:
        return None
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("Telemetry line must decode to a JSON object")

    data.setdefault("machine_id", default_machine_id)
    data.setdefault("motor_id", default_motor_id)
    data.setdefault("recording_id", f"ser-{uuid4()}")
    return data


def build_simulated_payload(
    *,
    step_index: int,
    machine_id: str,
    motor_id: str,
    fault: str,
    severity: float,
    sample_rate_hz: float,
    rpm: float,
    load_pct: float,
    window_size: int,
) -> dict[str, Any]:
    """Generate a realistic BLDC telemetry payload for hardware-free testing."""
    if fault == "cycle":
        fault_type, cycle_sev = _CYCLE_SEQUENCE[step_index % len(_CYCLE_SEQUENCE)]
        eff_severity = cycle_sev
    else:
        fault_type = fault  # type: ignore[assignment]
        eff_severity = severity

    vib, curr, temp_c = inject_fault(
        fault_type=fault_type,
        severity=eff_severity,
        snr_db=24.0,
        drift_offset=0.0,
        sample_rate_hz=sample_rate_hz,
        rpm=rpm,
        load_pct=load_pct,
        window_size=window_size,
        seed=1000 + step_index,
    )
    return {
        "machine_id": machine_id,
        "recording_id": f"sim-{uuid4()}",
        "motor_id": motor_id,
        "sample_rate_hz": sample_rate_hz,
        "rpm": rpm,
        "load_pct": load_pct,
        "label": fault_type,
        "vibration_g": vib,
        "current_a": curr,
        "temperature_c": temp_c,
    }


def post_payload(
    client: httpx.Client,
    url: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """POST a telemetry window to /api/v1/recordings/ingest and return parsed JSON."""
    resp = client.post(url, json=payload)
    resp.raise_for_status()
    return resp.json()


def run_bridge(
    args: argparse.Namespace,
    http_client: Optional[httpx.Client] = None,
) -> int:
    """Execute serial reading or simulated telemetry streaming loop."""
    own_client = http_client is None
    client = http_client or httpx.Client(timeout=args.timeout)
    sent = 0

    try:
        if args.simulate:
            step = 0
            while args.count <= 0 or sent < args.count:
                payload = build_simulated_payload(
                    step_index=step,
                    machine_id=args.machine_id,
                    motor_id=args.motor_id,
                    fault=args.fault,
                    severity=args.severity,
                    sample_rate_hz=args.sample_rate,
                    rpm=args.rpm,
                    load_pct=args.load_pct,
                    window_size=args.window_size,
                )
                result = post_payload(client, args.url, payload)
                dash = result["dashboard"]
                print(
                    f"[SIM #{sent + 1}] machine={dash['machine_id']} "
                    f"tier={dash['tier']} ({dash['status']}) "
                    f"fault={dash['fault']} conf={dash['confidence']:.2f} "
                    f"score={dash['anomaly_score']:.2f}"
                )
                sent += 1
                step += 1
                if (args.count <= 0 or sent < args.count) and args.interval > 0:
                    time.sleep(args.interval)
            return sent

        try:
            import serial  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError(
                "pyserial is required for hardware serial mode. "
                "Install it with `pip install pyserial` or pass `--simulate`."
            ) from exc

        with serial.Serial(args.port, args.baud, timeout=1.0) as ser:
            print(f"Listening on {args.port} @ {args.baud} baud -> {args.url}")
            while args.count <= 0 or sent < args.count:
                raw_bytes = ser.readline()
                if not raw_bytes:
                    continue
                line = raw_bytes.decode("utf-8", errors="replace")
                try:
                    payload = parse_serial_line(
                        line,
                        default_machine_id=args.machine_id,
                        default_motor_id=args.motor_id,
                    )
                except (json.JSONDecodeError, ValueError) as err:
                    print(f"[WARN] Skipping malformed serial line: {err}", file=sys.stderr)
                    continue
                if payload is None:
                    continue
                result = post_payload(client, args.url, payload)
                dash = result["dashboard"]
                print(
                    f"[SERIAL #{sent + 1}] machine={dash['machine_id']} "
                    f"tier={dash['tier']} ({dash['status']}) "
                    f"fault={dash['fault']}"
                )
                sent += 1
        return sent
    finally:
        if own_client:
            client.close()


def build_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Bridge ESP32/Arduino USB Serial NDJSON telemetry to FastAPI /ingest."
    )
    parser.add_argument("--port", default="COM3", help="Serial port (e.g. COM3, /dev/ttyUSB0)")
    parser.add_argument("--baud", type=int, default=115200, help="Serial baud rate")
    parser.add_argument(
        "--url",
        default="http://localhost:8000/api/v1/recordings/ingest",
        help="Target /ingest endpoint URL",
    )
    parser.add_argument("--machine-id", default="bldc-01", help="Default machine_id")
    parser.add_argument("--motor-id", default="motor-01", help="Default motor_id")
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Generate simulated BLDC telemetry instead of reading USB serial",
    )
    parser.add_argument(
        "--fault",
        default="cycle",
        choices=[
            "cycle",
            "healthy",
            "imbalance",
            "misalignment",
            "bearing_wear",
            "demagnetization",
            "combined",
        ],
        help="Fault mode when using --simulate",
    )
    parser.add_argument("--severity", type=float, default=0.55, help="Simulated fault severity (0.05-1.0)")
    parser.add_argument("--sample-rate", type=float, default=1000.0, help="Sample rate in Hz")
    parser.add_argument("--rpm", type=float, default=1500.0, help="Simulated motor RPM")
    parser.add_argument("--load-pct", type=float, default=55.0, help="Simulated motor load percentage")
    parser.add_argument("--window-size", type=int, default=256, help="Samples per simulated window")
    parser.add_argument(
        "--count",
        type=int,
        default=0,
        help="Number of windows to send (0 = stream continuously)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Delay in seconds between simulated windows",
    )
    parser.add_argument("--timeout", type=float, default=10.0, help="HTTP request timeout in seconds")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        run_bridge(args)
    except KeyboardInterrupt:
        print("\nStopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
