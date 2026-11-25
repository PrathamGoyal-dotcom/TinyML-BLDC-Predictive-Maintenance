#!/usr/bin/env python3
"""DUDU-BLDC 28-feature tabular baseline CLI with leakage-free recording_id-grouped K-fold CV."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.ml.baseline import (  # noqa: E402
    evaluate_dudu_baseline,
    generate_synthetic_bldc_dataset,
    load_csv_dataset,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate DUDU-BLDC 28-feature baseline with recording_id-grouped K-fold CV."
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Path to exported CSV from /api/v1/recordings/export/csv (defaults to synthetic generator)",
    )
    parser.add_argument(
        "--n-recordings",
        type=int,
        default=60,
        help="Number of synthetic physical recordings (used when --csv is omitted)",
    )
    parser.add_argument(
        "--windows-per-recording",
        type=int,
        default=3,
        help="Windows per physical recording_id in synthetic mode",
    )
    parser.add_argument("--splits", type=int, default=5, help="Number of grouped CV folds")
    parser.add_argument("--snr-db", type=float, default=22.0, help="Synthetic AWGN SNR (dB)")
    parser.add_argument("--model", choices=["mlp", "rf"], default="mlp", help="Tabular classifier")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--json", action="store_true", help="Output raw JSON report")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.csv:
        X, y, groups = load_csv_dataset(args.csv)
        source = "csv"
    else:
        X, y, groups, _ = generate_synthetic_bldc_dataset(
            n_recordings=args.n_recordings,
            windows_per_recording=args.windows_per_recording,
            snr_db=args.snr_db,
            seed=args.seed,
        )
        source = "synthetic"

    report = evaluate_dudu_baseline(
        X,
        y,
        groups,
        source=source,  # type: ignore[arg-type]
        n_splits=args.splits,
        seed=args.seed,
        model_type=args.model,
    )

    if args.json:
        print(report.model_dump_json(indent=2))
        return 0

    print(f"=== {report.benchmark_name} ({report.model_type.upper()}) ===")
    print(f"Source            : {report.source} ({report.n_samples} windows, {report.n_recordings} recording_id groups)")
    print(f"CV Protocol       : {report.cv_protocol} [leakage_verified={report.leakage_verified}]")
    print(f"Overall Accuracy  : {report.overall_accuracy * 100:.2f}%")
    print(f"Balanced Accuracy : {report.balanced_accuracy * 100:.2f}%")
    if report.quantized_int8_accuracy is not None:
        print(f"Int8 Quant Acc    : {report.quantized_int8_accuracy * 100:.2f}%")
    print(f"Avg Latency       : {report.avg_inference_latency_ms:.4f} ms / window")
    print(
        f"Model Footprint   : {report.model_footprint_kb.float32_kb:.2f} KB (float32) | "
        f"{report.model_footprint_kb.int8_quantized_kb:.2f} KB (int8) "
        f"[{report.model_footprint_kb.parameter_count} params]"
    )
    print("Per-Class F1      :", json.dumps(report.per_class_f1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
