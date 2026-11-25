"""Pydantic v2 request/response schemas and canonical feature definitions."""

from __future__ import annotations

import math
from typing import Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator

FaultType = Literal[
    "healthy",
    "imbalance",
    "misalignment",
    "bearing_wear",
    "demagnetization",
    "combined",
]

HealthStatus = Literal["healthy", "needs_attention", "about_to_fail"]
LanguageCode = Literal["en", "hi"]

SIGNAL_FEATURE_SUFFIXES: tuple[str, ...] = (
    "mean",
    "std",
    "max",
    "rms",
    "peak_to_peak",
    "skewness",
    "kurtosis",
    "crest_factor",
    "spectral_centroid",
    "spectrum_area",
    "hf_bearing_energy",
    "harmonic_1x",
    "harmonic_2x",
    "harmonic_3x",
)

FEATURE_NAMES: tuple[str, ...] = tuple(
    [f"vib_{name}" for name in SIGNAL_FEATURE_SUFFIXES]
    + [f"curr_{name}" for name in SIGNAL_FEATURE_SUFFIXES]
)

assert len(FEATURE_NAMES) == 28


def _validate_finite_series(values: list[float], field_name: str) -> list[float]:
    for val in values:
        if not math.isfinite(val):
            raise ValueError(f"{field_name} must contain only finite numbers")
    return values


class InferenceResult(BaseModel):
    """Compact TinyML classification and anomaly scoring output."""

    health_tier: int = Field(..., ge=0, le=2)
    health_status: HealthStatus
    predicted_fault: FaultType
    confidence: float = Field(..., ge=0.0, le=1.0)
    anomaly_score: float = Field(..., ge=0.0, le=1.0)


class DashboardResponse(BaseModel):
    """Ultra-lightweight payload tailored for low-bandwidth mobile dashboards."""

    machine_id: str
    tier: int = Field(..., ge=0, le=2)
    status: HealthStatus
    fault: FaultType
    confidence: float = Field(..., ge=0.0, le=1.0)
    anomaly_score: float = Field(..., ge=0.0, le=1.0)
    vibration: float = Field(default=0.0, ge=0.0)
    current: float = Field(default=0.0, ge=0.0)
    sound: float = Field(default=0.0, ge=0.0)
    degradation: float = Field(default=0.0, ge=0.0, le=100.0)
    rpm: float
    temperature_c: float
    language: LanguageCode
    headline: str
    action_hint: str
    updated_at: str


class RecordingIngestRequest(BaseModel):
    """Telemetry window posted by MCU serial bridge or edge gateway."""

    machine_id: str = Field(..., min_length=1, max_length=64)
    recording_id: str = Field(
        default_factory=lambda: str(uuid4()), min_length=1, max_length=64
    )
    motor_id: str = Field(..., min_length=1, max_length=64)
    sample_rate_hz: float = Field(..., gt=0.0, le=100000.0)
    rpm: float = Field(..., ge=0.0, le=50000.0)
    load_pct: float = Field(..., ge=0.0, le=200.0)
    label: Optional[str] = Field(default=None, max_length=64)
    vibration_g: list[float] = Field(..., min_length=8, max_length=16384)
    current_a: Optional[list[float]] = Field(default=None, max_length=16384)
    sound_db: Optional[float] = Field(default=None, ge=0.0, le=160.0)
    temperature_c: float = Field(..., ge=-40.0, le=200.0)

    @field_validator("vibration_g")
    @classmethod
    def check_vibration(cls, v: list[float]) -> list[float]:
        return _validate_finite_series(v, "vibration_g")

    @field_validator("current_a")
    @classmethod
    def check_current(cls, v: Optional[list[float]]) -> Optional[list[float]]:
        if v is None or len(v) == 0:
            return None
        if len(v) < 8:
            raise ValueError("current_a must have at least 8 samples when provided")
        return _validate_finite_series(v, "current_a")


class RecordingIngestResponse(BaseModel):
    """Response returned after ingesting, extracting features, and running inference."""

    recording_id: str
    machine_id: str
    motor_id: str
    timestamp: str
    features: dict[str, float]
    inference: InferenceResult
    dashboard: DashboardResponse


class RecordingRecord(BaseModel):
    """Stored recording record returned by GET /api/v1/recordings/."""

    recording_id: str
    machine_id: str
    motor_id: str
    timestamp: str
    sample_rate_hz: float
    rpm: float
    load_pct: float
    temperature_c: float
    label: Optional[str]
    is_synthetic: bool
    sample_count: int
    features: dict[str, float]
    inference: InferenceResult
    vibration_g: Optional[list[float]] = None
    current_a: Optional[list[float]] = None


class RecordingListResponse(BaseModel):
    """Paginated list of stored recordings."""

    total: int
    limit: int
    offset: int
    items: list[RecordingRecord]


class InjectRequest(BaseModel):
    """Controlled BLDC fault injection configuration."""

    machine_id: str = Field(default="bldc-01", min_length=1, max_length=64)
    motor_id: str = Field(default="motor-01", min_length=1, max_length=64)
    recording_id: Optional[str] = Field(default=None, min_length=1, max_length=64)
    fault_type: FaultType
    severity: float = Field(default=0.5, ge=0.05, le=1.0)
    snr_db: float = Field(default=25.0, ge=-10.0, le=80.0)
    drift_offset: float = Field(default=0.0, ge=-10.0, le=10.0)
    sample_rate_hz: float = Field(default=1000.0, gt=0.0, le=100000.0)
    rpm: float = Field(default=1500.0, gt=0.0, le=50000.0)
    load_pct: float = Field(default=50.0, ge=0.0, le=200.0)
    temperature_c: Optional[float] = Field(default=None, ge=-40.0, le=200.0)
    window_size: int = Field(default=256, ge=16, le=8192)
    base_vibration_g: Optional[list[float]] = Field(
        default=None, min_length=8, max_length=16384
    )
    base_current_a: Optional[list[float]] = Field(
        default=None, min_length=8, max_length=16384
    )
    sound_db: Optional[float] = Field(default=None, ge=0.0, le=160.0)
    persist: bool = Field(default=False)
    seed: Optional[int] = Field(default=None)

    @field_validator("base_vibration_g")
    @classmethod
    def check_base_vibration(cls, v: Optional[list[float]]) -> Optional[list[float]]:
        if v is None:
            return None
        return _validate_finite_series(v, "base_vibration_g")

    @field_validator("base_current_a")
    @classmethod
    def check_base_current(cls, v: Optional[list[float]]) -> Optional[list[float]]:
        if v is None:
            return None
        return _validate_finite_series(v, "base_current_a")


class InjectResponse(BaseModel):
    """Result of controlled fault injection + TinyML inference + dashboard update."""

    recording_id: str
    machine_id: str
    motor_id: str
    persisted: bool
    injected_fault: FaultType
    severity: float
    snr_db: float
    drift_offset: float
    sample_rate_hz: float
    rpm: float
    temperature_c: float
    features: dict[str, float]
    inference: InferenceResult
    dashboard: DashboardResponse
    vibration_g: list[float]
    current_a: list[float]


class ModelFootprintKB(BaseModel):
    """Memory footprint of the trained tabular TinyML classifier."""

    float32_kb: float
    int8_quantized_kb: float
    parameter_count: int


class BaselineEvaluateRequest(BaseModel):
    """Configuration for leakage-free recording_id-grouped K-fold baseline evaluation."""

    source: Literal["synthetic", "database", "csv"] = Field(default="synthetic")
    csv_path: Optional[str] = Field(default=None)
    n_recordings: int = Field(default=60, ge=12, le=1000)
    windows_per_recording: int = Field(default=3, ge=1, le=20)
    n_splits: int = Field(default=5, ge=2, le=10)
    snr_db: float = Field(default=22.0, ge=-10.0, le=80.0)
    seed: int = Field(default=42)
    model_type: Literal["mlp", "rf"] = Field(default="mlp")


class BaselineEvaluateResponse(BaseModel):
    """Evaluation report for the DUDU-BLDC 28-feature tabular baseline."""

    benchmark_name: str
    model_type: Literal["mlp", "rf"]
    source: Literal["synthetic", "database", "csv"]
    cv_protocol: str
    leakage_verified: bool
    n_samples: int
    n_recordings: int
    n_splits: int
    overall_accuracy: float
    balanced_accuracy: float
    quantized_int8_accuracy: Optional[float] = None
    per_class_f1: dict[str, float]
    avg_inference_latency_ms: float
    model_footprint_kb: ModelFootprintKB
