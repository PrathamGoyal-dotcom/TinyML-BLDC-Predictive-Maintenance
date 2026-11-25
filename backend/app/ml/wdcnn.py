#!/usr/bin/env python3
"""WDCNN (Wide First-Layer Kernel Deep 1D-CNN) reference model in PyTorch.

Implements the 5-stage 1D convolutional architecture (64x1 wide first-layer kernel +
four 3x1 stages, all with BatchNorm1d, ReLU, and MaxPool1d) for raw 1D vibration
fault diagnosis and evaluates accuracy under noisy (AWGN) test conditions.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.ml.baseline import FAULT_CLASSES, generate_synthetic_bldc_dataset  # noqa: E402


class WDCNN(nn.Module):
    """5-stage 1D-CNN with a 64x1 wide first-layer kernel and BatchNorm."""

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 6,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.stage1 = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=64, stride=4, padding=30, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.stage2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.stage3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.stage4 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.stage5 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.pool(x)
        return self.classifier(x)


@dataclass
class WDCNNBenchmarkReport:
    benchmark_name: str
    cv_protocol: str
    leakage_verified: bool
    n_train_samples: int
    n_test_samples: int
    epochs: int
    window_size: int
    parameter_count: int
    float32_kb: float
    int8_quantized_kb: float
    avg_inference_latency_ms: float
    clean_accuracy: float
    clean_balanced_accuracy: float
    per_class_f1: dict[str, float]
    snr_accuracy: dict[str, float]


def add_awgn_to_windows(
    signals: np.ndarray,
    snr_db: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Inject AWGN at a target SNR (dB) onto raw 1D vibration windows."""
    noisy = np.empty_like(signals, dtype=np.float32)
    for idx, sig in enumerate(signals):
        ac = sig - np.mean(sig)
        rms_ac = float(np.sqrt(np.mean(ac * ac)))
        base_rms = max(rms_ac, 0.02)
        noise_std = base_rms * (10.0 ** (-float(snr_db) / 20.0))
        noisy[idx] = sig + rng.normal(0.0, noise_std, size=len(sig)).astype(np.float32)
    return noisy


def train_and_evaluate_wdcnn(
    *,
    n_recordings: int = 90,
    windows_per_recording: int = 4,
    window_size: int = 256,
    epochs: int = 25,
    batch_size: int = 32,
    lr: float = 1e-3,
    train_snr_db: float = 22.0,
    eval_snr_levels: tuple[float, ...] = (22.0, 10.0, 5.0, 0.0),
    seed: int = 42,
) -> WDCNNBenchmarkReport:
    """Train WDCNN on leakage-free recording_id split and evaluate under AWGN noise."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    _, y_str, groups, raw_vib = generate_synthetic_bldc_dataset(
        n_recordings=n_recordings,
        windows_per_recording=windows_per_recording,
        snr_db=train_snr_db,
        window_size=window_size,
        seed=seed,
        include_raw_signals=True,
    )
    assert raw_vib is not None

    class_to_idx = {name: i for i, name in enumerate(FAULT_CLASSES)}
    y_idx = np.asarray([class_to_idx[str(lbl)] for lbl in y_str], dtype=np.int64)

    # First fold of StratifiedGroupKFold gives a clean 80/20 recording_id-grouped split
    splitter = StratifiedGroupKFold(n_splits=5)
    train_idx, test_idx = next(splitter.split(raw_vib, y_idx, groups))
    if set(groups[train_idx].tolist()) & set(groups[test_idx].tolist()):
        raise RuntimeError("Data leakage detected between train and test recording_id groups")

    X_train_raw = raw_vib[train_idx]
    y_train = y_idx[train_idx]
    X_test_raw = raw_vib[test_idx]
    y_test = y_idx[test_idx]

    # Remove per-window sensor DC drift while preserving AC amplitude scale globally
    X_train_ac = X_train_raw - np.mean(X_train_raw, axis=1, keepdims=True)
    X_test_ac = X_test_raw - np.mean(X_test_raw, axis=1, keepdims=True)

    train_scale = float(np.std(X_train_ac)) + 1e-6
    X_train_norm = (X_train_ac / train_scale).astype(np.float32)

    device = torch.device("cpu")
    model = WDCNN(in_channels=1, num_classes=len(FAULT_CLASSES)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    train_ds = TensorDataset(
        torch.from_numpy(X_train_norm).unsqueeze(1),
        torch.from_numpy(y_train),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model.train()
    for _ in range(epochs):
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            # Light AWGN data augmentation during training improves anti-noise robustness
            aug_noise = torch.randn_like(xb) * 0.05
            optimizer.zero_grad()
            logits = model(xb + aug_noise)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

    model.eval()
    # Evaluate on nominal test split + latency
    X_test_norm = torch.from_numpy((X_test_ac / train_scale).astype(np.float32)).unsqueeze(1)
    with torch.no_grad():
        # Warmup
        _ = model(X_test_norm[: min(8, len(X_test_norm))])
        t0 = time.perf_counter()
        logits = model(X_test_norm)
        elapsed_s = time.perf_counter() - t0
        preds = torch.argmax(logits, dim=1).cpu().numpy()

    avg_latency_ms = round((elapsed_s * 1000.0) / max(len(X_test_norm), 1), 4)
    clean_acc = round(float(accuracy_score(y_test, preds)), 4)
    clean_bal_acc = round(float(balanced_accuracy_score(y_test, preds)), 4)
    f1_vals = f1_score(
        y_test,
        preds,
        labels=list(range(len(FAULT_CLASSES))),
        average=None,
        zero_division=0,
    )
    per_class_f1 = {
        cls_name: round(float(score), 4)
        for cls_name, score in zip(FAULT_CLASSES, f1_vals)
    }

    snr_accuracy: dict[str, float] = {}
    with torch.no_grad():
        for snr_db in eval_snr_levels:
            noisy_raw = add_awgn_to_windows(X_test_ac, snr_db=snr_db, rng=rng)
            noisy_norm = torch.from_numpy((noisy_raw / train_scale).astype(np.float32)).unsqueeze(1)
            noisy_preds = torch.argmax(model(noisy_norm), dim=1).cpu().numpy()
            snr_accuracy[f"{snr_db:g}dB"] = round(
                float(accuracy_score(y_test, noisy_preds)), 4
            )

    param_count = int(sum(p.numel() for p in model.parameters()))
    float32_kb = round((param_count * 4) / 1024.0, 2)
    int8_kb = round((param_count * 1) / 1024.0, 2)

    return WDCNNBenchmarkReport(
        benchmark_name="WDCNN 5-Stage 1D-CNN (64x1 Kernel + BatchNorm)",
        cv_protocol="StratifiedGroupKFold(n_splits=5, fold=0, group=recording_id)",
        leakage_verified=True,
        n_train_samples=int(len(X_train_norm)),
        n_test_samples=int(len(X_test_norm)),
        epochs=epochs,
        window_size=window_size,
        parameter_count=param_count,
        float32_kb=float32_kb,
        int8_quantized_kb=int8_kb,
        avg_inference_latency_ms=avg_latency_ms,
        clean_accuracy=clean_acc,
        clean_balanced_accuracy=clean_bal_acc,
        per_class_f1=per_class_f1,
        snr_accuracy=snr_accuracy,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the WDCNN 1D-CNN reference model under AWGN noise."
    )
    parser.add_argument("--n-recordings", type=int, default=90, help="Synthetic recording_id groups")
    parser.add_argument("--windows-per-recording", type=int, default=4, help="Windows per recording")
    parser.add_argument("--window-size", type=int, default=256, help="1D signal window length")
    parser.add_argument("--epochs", type=int, default=25, help="Training epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--snr-levels",
        type=float,
        nargs="+",
        default=[22.0, 10.0, 5.0, 0.0],
        help="AWGN test SNR levels in dB",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON report")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    report = train_and_evaluate_wdcnn(
        n_recordings=args.n_recordings,
        windows_per_recording=args.windows_per_recording,
        window_size=args.window_size,
        epochs=args.epochs,
        eval_snr_levels=tuple(args.snr_levels),
        seed=args.seed,
    )
    if args.json:
        print(json.dumps(asdict(report), indent=2))
        return 0

    print(f"=== {report.benchmark_name} ===")
    print(f"Split Protocol    : {report.cv_protocol} [leakage_verified={report.leakage_verified}]")
    print(f"Train / Test      : {report.n_train_samples} / {report.n_test_samples} windows (len={report.window_size}, epochs={report.epochs})")
    print(f"Clean Accuracy    : {report.clean_accuracy * 100:.2f}% (Balanced: {report.clean_balanced_accuracy * 100:.2f}%)")
    print(f"Avg Latency       : {report.avg_inference_latency_ms:.4f} ms / window")
    print(f"Model Footprint   : {report.float32_kb:.2f} KB (float32) | {report.int8_quantized_kb:.2f} KB (int8) [{report.parameter_count} params]")
    print("AWGN SNR Accuracy :", json.dumps(report.snr_accuracy))
    print("Per-Class F1      :", json.dumps(report.per_class_f1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
