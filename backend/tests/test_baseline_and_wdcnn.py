"""Tests for DUDU-BLDC 28-feature baseline (API + CLI) and WDCNN 1D-CNN model."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
import torch
import torch.nn as nn

from app.ml.wdcnn import WDCNN, train_and_evaluate_wdcnn
from scripts.baseline_eval import main as baseline_cli_main


def test_baseline_evaluate_endpoint_synthetic_mlp_and_rf(client: TestClient) -> None:
    resp = client.post(
        "/api/v1/baseline/evaluate",
        json={
            "source": "synthetic",
            "n_recordings": 36,
            "windows_per_recording": 3,
            "n_splits": 3,
            "snr_db": 22.0,
            "model_type": "mlp",
            "seed": 42,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["leakage_verified"] is True
    assert "recording_id" in data["cv_protocol"]
    assert data["n_samples"] == 108
    assert data["n_recordings"] == 36
    assert data["n_splits"] == 3
    assert data["overall_accuracy"] >= 0.80
    assert data["balanced_accuracy"] >= 0.80
    assert data["quantized_int8_accuracy"] is not None
    assert data["avg_inference_latency_ms"] >= 0.0
    assert data["model_footprint_kb"]["int8_quantized_kb"] < data["model_footprint_kb"]["float32_kb"]
    assert len(data["per_class_f1"]) == 6

    rf_resp = client.post(
        "/api/v1/baseline/evaluate",
        json={
            "source": "synthetic",
            "n_recordings": 36,
            "windows_per_recording": 2,
            "n_splits": 3,
            "model_type": "rf",
            "seed": 42,
        },
    )
    assert rf_resp.status_code == 200
    assert rf_resp.json()["model_type"] == "rf"
    assert rf_resp.json()["overall_accuracy"] >= 0.80


def test_baseline_evaluate_from_exported_csv_and_database(
    client: TestClient, tmp_path: Path
) -> None:
    # Seed 12 labeled recordings across 6 fault classes via /api/v1/inject/ with persist=True
    faults = [
        "healthy",
        "imbalance",
        "misalignment",
        "bearing_wear",
        "demagnetization",
        "combined",
    ]
    for idx in range(12):
        fault = faults[idx % len(faults)]
        r = client.post(
            "/api/v1/inject/",
            json={
                "machine_id": f"m-{idx}",
                "recording_id": f"db-rec-{idx:02d}",
                "fault_type": fault,
                "severity": 0.15 if fault == "healthy" else 0.75,
                "persist": True,
                "seed": idx + 10,
            },
        )
        assert r.status_code == 200

    # Export CSV from backend and evaluate via both API (source='csv') and CLI (--csv)
    csv_resp = client.get("/api/v1/recordings/export/csv")
    assert csv_resp.status_code == 200
    csv_file = tmp_path / "exported_bldc.csv"
    csv_file.write_text(csv_resp.text, encoding="utf-8")

    eval_csv = client.post(
        "/api/v1/baseline/evaluate",
        json={"source": "csv", "csv_path": str(csv_file), "n_splits": 2},
    )
    assert eval_csv.status_code == 200
    assert eval_csv.json()["n_samples"] == 12
    assert eval_csv.json()["leakage_verified"] is True

    eval_db = client.post(
        "/api/v1/baseline/evaluate",
        json={"source": "database", "n_splits": 2},
    )
    assert eval_db.status_code == 200
    assert eval_db.json()["n_samples"] == 12

    rc = baseline_cli_main(["--csv", str(csv_file), "--splits", "2", "--json"])
    assert rc == 0


def test_wdcnn_architecture_and_noisy_awgn_evaluation() -> None:
    model = WDCNN(in_channels=1, num_classes=6)
    # Verify 5 conv stages with 64x1 wide first-layer kernel and BatchNorm1d
    stages = [model.stage1, model.stage2, model.stage3, model.stage4, model.stage5]
    assert len(stages) == 5
    first_conv = model.stage1[0]
    assert isinstance(first_conv, nn.Conv1d)
    assert first_conv.kernel_size == (64,)
    for stage in stages:
        assert any(isinstance(layer, nn.BatchNorm1d) for layer in stage)

    for length in (256, 512):
        out = model(torch.randn(4, 1, length))
        assert out.shape == (4, 6)

    report = train_and_evaluate_wdcnn(
        n_recordings=36,
        windows_per_recording=3,
        window_size=256,
        epochs=6,
        eval_snr_levels=(22.0, 5.0, 0.0),
        seed=42,
    )
    assert report.leakage_verified is True
    assert report.clean_accuracy >= 0.65
    assert set(report.snr_accuracy.keys()) == {"22dB", "5dB", "0dB"}
    assert report.float32_kb > report.int8_quantized_kb > 0.0
