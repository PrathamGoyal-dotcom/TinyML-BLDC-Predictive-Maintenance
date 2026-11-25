"""DUDU-BLDC 28-feature tabular baseline with leakage-free recording_id-grouped K-fold CV."""

from __future__ import annotations

import csv
from pathlib import Path
import time
from typing import Literal, Optional, get_args

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

from app.db.repository import RecordingsRepository
from app.ml.fault_injector import inject_fault
from app.ml.features import extract_28_features
from app.schemas import (
    FEATURE_NAMES,
    BaselineEvaluateResponse,
    FaultType,
    ModelFootprintKB,
)

FAULT_CLASSES: tuple[FaultType, ...] = get_args(FaultType)


def generate_synthetic_bldc_dataset(
    *,
    n_recordings: int = 60,
    windows_per_recording: int = 3,
    snr_db: float = 22.0,
    sample_rate_hz: float = 1000.0,
    window_size: int = 256,
    seed: int = 42,
    include_raw_signals: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Generate a synthetic BLDC dataset with multiple windows per physical recording_id."""
    rng = np.random.default_rng(seed)
    X_rows: list[list[float]] = []
    y_rows: list[str] = []
    groups: list[str] = []
    raw_vib_rows: list[list[float]] = []

    n_classes = len(FAULT_CLASSES)
    for rec_idx in range(n_recordings):
        fault_type = FAULT_CLASSES[rec_idx % n_classes]
        rec_id = f"rec-{rec_idx:04d}"

        # Recording-level physical operating conditions
        base_rpm = float(rng.uniform(1200.0, 1800.0))
        base_load = float(rng.uniform(30.0, 80.0))
        base_sev = (
            float(rng.uniform(0.05, 0.20))
            if fault_type == "healthy"
            else float(rng.uniform(0.28, 0.95))
        )
        base_drift = float(rng.uniform(-0.25, 0.25))
        rec_snr = float(snr_db + rng.uniform(-3.0, 3.0))

        for win_idx in range(windows_per_recording):
            win_seed = int(seed * 10000 + rec_idx * 100 + win_idx)
            rpm = base_rpm + float(rng.uniform(-25.0, 25.0))
            sev = max(0.05, min(1.0, base_sev + float(rng.uniform(-0.04, 0.04))))

            vib, curr, _ = inject_fault(
                fault_type=fault_type,
                severity=sev,
                snr_db=rec_snr,
                drift_offset=base_drift,
                sample_rate_hz=sample_rate_hz,
                rpm=rpm,
                load_pct=base_load,
                window_size=window_size,
                seed=win_seed,
            )
            feats = extract_28_features(
                vibration_g=vib,
                current_a=curr,
                sample_rate_hz=sample_rate_hz,
                rpm=rpm,
            )
            X_rows.append([feats[name] for name in FEATURE_NAMES])
            y_rows.append(fault_type)
            groups.append(rec_id)
            if include_raw_signals:
                raw_vib_rows.append(vib)

    X = np.asarray(X_rows, dtype=np.float32)
    y = np.asarray(y_rows, dtype=object)
    g = np.asarray(groups, dtype=object)
    raw = np.asarray(raw_vib_rows, dtype=np.float32) if include_raw_signals else None
    return X, y, g, raw


def load_csv_dataset(csv_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load 28 features, ground-truth labels, and recording_id groups from exported CSV."""
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    X_rows: list[list[float]] = []
    y_rows: list[str] = []
    groups: list[str] = []

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("CSV file is empty or missing headers")
        missing = [f for f in FEATURE_NAMES if f not in reader.fieldnames]
        if missing:
            raise ValueError(f"CSV missing required feature columns: {missing[:5]}")
        if "recording_id" not in reader.fieldnames or "label" not in reader.fieldnames:
            raise ValueError("CSV must contain 'recording_id' and 'label' columns")

        for row in reader:
            label = (row.get("label") or "").strip()
            rec_id = (row.get("recording_id") or "").strip()
            if not label or not rec_id:
                continue
            X_rows.append([float(row[name]) for name in FEATURE_NAMES])
            y_rows.append(label)
            groups.append(rec_id)

    if not X_rows:
        raise ValueError("CSV contains no labeled rows with recording_id")

    return (
        np.asarray(X_rows, dtype=np.float32),
        np.asarray(y_rows, dtype=object),
        np.asarray(groups, dtype=object),
    )


def load_db_dataset(
    repo: RecordingsRepository,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load labeled 28-feature windows and recording_id groups from SQLite."""
    rows = repo.iter_export_rows(include_synthetic=True)
    X_rows: list[list[float]] = []
    y_rows: list[str] = []
    groups: list[str] = []

    for row in rows:
        label = (row["label"] or "").strip()
        if not label:
            continue
        X_rows.append([float(row[name]) for name in FEATURE_NAMES])
        y_rows.append(label)
        groups.append(str(row["recording_id"]))

    if not X_rows:
        raise ValueError("Database contains no labeled recordings for evaluation")

    return (
        np.asarray(X_rows, dtype=np.float32),
        np.asarray(y_rows, dtype=object),
        np.asarray(groups, dtype=object),
    )


def _quantized_mlp_predict(
    X_scaled: np.ndarray,
    coefs: list[np.ndarray],
    intercepts: list[np.ndarray],
) -> np.ndarray:
    """Simulate int8 symmetric weight quantization forward pass for TinyML footprint/fidelity."""
    h = X_scaled.astype(np.float32)
    n_layers = len(coefs)
    for idx, (w, b) in enumerate(zip(coefs, intercepts)):
        max_abs = float(np.max(np.abs(w)))
        scale = max_abs / 127.0 if max_abs > 1e-12 else 1.0
        w_q = np.clip(np.round(w / scale), -127, 127).astype(np.int8)
        w_deq = w_q.astype(np.float32) * scale
        h = h @ w_deq + b.astype(np.float32)
        if idx < n_layers - 1:
            h = np.maximum(h, 0.0)
    return np.argmax(h, axis=1)


def _compute_footprint(
    model: object,
    model_type: Literal["mlp", "rf"],
    n_features: int,
) -> ModelFootprintKB:
    if model_type == "mlp":
        assert isinstance(model, MLPClassifier)
        n_params = sum(w.size for w in model.coefs_) + sum(b.size for b in model.intercepts_)
        scaler_params = 2 * n_features
        total_params = int(n_params + scaler_params)
        float32_kb = round((total_params * 4) / 1024.0, 3)
        # Weights in int8 (1B) + biases/scaler in float32 (4B) + per-layer scales
        n_weights = sum(w.size for w in model.coefs_)
        n_biases = sum(b.size for b in model.intercepts_) + scaler_params
        int8_bytes = n_weights * 1 + n_biases * 4 + len(model.coefs_) * 4
        int8_kb = round(int8_bytes / 1024.0, 3)
        return ModelFootprintKB(
            float32_kb=float32_kb,
            int8_quantized_kb=int8_kb,
            parameter_count=total_params,
        )

    assert isinstance(model, RandomForestClassifier)
    total_nodes = int(sum(est.tree_.node_count for est in model.estimators_))
    # Per node in C/TinyML struct: feature_idx (int16/int32) + threshold (f32) + left/right (2x int16) + leaf value
    float32_bytes = total_nodes * (4 + 4 + 4 + 4)
    int8_bytes = total_nodes * (1 + 1 + 2 + 2)
    return ModelFootprintKB(
        float32_kb=round(float32_bytes / 1024.0, 3),
        int8_quantized_kb=round(int8_bytes / 1024.0, 3),
        parameter_count=total_nodes,
    )


def evaluate_dudu_baseline(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    source: Literal["synthetic", "database", "csv"],
    n_splits: int = 5,
    seed: int = 42,
    model_type: Literal["mlp", "rf"] = "mlp",
) -> BaselineEvaluateResponse:
    """Run leakage-free recording_id-grouped K-fold CV on the 28-feature tabular schema."""
    unique_groups = np.unique(groups)
    n_recordings = int(len(unique_groups))
    if n_recordings < 2:
        raise ValueError("At least 2 distinct recording_id groups are required for K-fold CV")

    eff_splits = min(int(n_splits), n_recordings)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    class_names = [str(c) for c in le.classes_]

    # Prefer StratifiedGroupKFold; fall back to GroupKFold if a class has too few groups
    min_class_groups = min(
        len(np.unique(groups[y_enc == cls_idx])) for cls_idx in np.unique(y_enc)
    )
    if min_class_groups >= eff_splits:
        splitter = StratifiedGroupKFold(n_splits=eff_splits)
        cv_protocol = f"StratifiedGroupKFold(n_splits={eff_splits}, group=recording_id)"
    else:
        splitter = GroupKFold(n_splits=eff_splits)
        cv_protocol = f"GroupKFold(n_splits={eff_splits}, group=recording_id)"

    oof_true: list[int] = []
    oof_pred: list[int] = []
    oof_quant_pred: list[int] = []
    total_infer_s = 0.0
    total_infer_samples = 0
    last_model: object = None

    for train_idx, test_idx in splitter.split(X, y_enc, groups):
        # Hard runtime assertion guaranteeing zero recording_id leakage across folds
        train_groups = set(groups[train_idx].tolist())
        test_groups = set(groups[test_idx].tolist())
        if train_groups & test_groups:
            raise RuntimeError("Data leakage detected: recording_id shared across train/test")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_enc[train_idx], y_enc[test_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        if model_type == "mlp":
            clf: object = MLPClassifier(
                hidden_layer_sizes=(32, 16),
                activation="relu",
                max_iter=350,
                random_state=seed,
            )
            clf.fit(X_train_s, y_train)
        else:
            clf = RandomForestClassifier(
                n_estimators=40,
                max_depth=8,
                random_state=seed,
            )
            clf.fit(X_train_s, y_train)

        t0 = time.perf_counter()
        preds = clf.predict(X_test_s)
        total_infer_s += time.perf_counter() - t0
        total_infer_samples += len(X_test_s)

        if model_type == "mlp":
            assert isinstance(clf, MLPClassifier)
            q_preds = _quantized_mlp_predict(X_test_s, clf.coefs_, clf.intercepts_)
            oof_quant_pred.extend(q_preds.tolist())

        oof_true.extend(y_test.tolist())
        oof_pred.extend(preds.tolist())
        last_model = clf

    y_true_arr = np.asarray(oof_true, dtype=int)
    y_pred_arr = np.asarray(oof_pred, dtype=int)

    overall_acc = round(float(accuracy_score(y_true_arr, y_pred_arr)), 4)
    balanced_acc = round(float(balanced_accuracy_score(y_true_arr, y_pred_arr)), 4)
    f1_vals = f1_score(
        y_true_arr,
        y_pred_arr,
        labels=list(range(len(class_names))),
        average=None,
        zero_division=0,
    )
    per_class_f1 = {
        name: round(float(score), 4) for name, score in zip(class_names, f1_vals)
    }
    avg_latency_ms = round(
        (total_infer_s * 1000.0) / max(total_infer_samples, 1), 4
    )
    quant_acc = (
        round(float(accuracy_score(y_true_arr, np.asarray(oof_quant_pred, dtype=int))), 4)
        if oof_quant_pred
        else None
    )
    footprint = _compute_footprint(last_model, model_type=model_type, n_features=X.shape[1])

    return BaselineEvaluateResponse(
        benchmark_name="DUDU-BLDC 28-Feature Tabular Baseline",
        model_type=model_type,
        source=source,
        cv_protocol=cv_protocol,
        leakage_verified=True,
        n_samples=int(len(y_true_arr)),
        n_recordings=n_recordings,
        n_splits=eff_splits,
        overall_accuracy=overall_acc,
        balanced_accuracy=balanced_acc,
        quantized_int8_accuracy=quant_acc,
        per_class_f1=per_class_f1,
        avg_inference_latency_ms=avg_latency_ms,
        model_footprint_kb=footprint,
    )
