"""Data Recording Pipeline endpoints (/api/v1/recordings/*)."""

from __future__ import annotations

import csv
import io
import sqlite3
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import StreamingResponse

from app.core.config import Settings, get_settings
from app.core.state import DashboardStateManager, utc_now_iso
from app.db.repository import RecordingsRepository
from app.ml.features import extract_28_features
from app.ml.inference import run_tinyml_inference
from app.schemas import (
    FEATURE_NAMES,
    RecordingIngestRequest,
    RecordingIngestResponse,
    RecordingListResponse,
)

router = APIRouter(prefix="/api/v1/recordings", tags=["Data Recording Pipeline"])


def get_repo(request: Request) -> RecordingsRepository:
    return request.app.state.repo


def get_state_manager(request: Request) -> DashboardStateManager:
    return request.app.state.dashboard_state


@router.post(
    "/ingest",
    response_model=RecordingIngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest raw vibration/current window, extract 28 features, and run TinyML inference",
)
def ingest_recording(
    payload: RecordingIngestRequest,
    language: str = Query(default="en", pattern="^(en|hi)$"),
    repo: RecordingsRepository = Depends(get_repo),
    state_mgr: DashboardStateManager = Depends(get_state_manager),
    settings: Settings = Depends(get_settings),
) -> RecordingIngestResponse:
    """Ingest a telemetry window from an MCU/gateway, extract features, store, and notify SSE."""
    features = extract_28_features(
        vibration_g=payload.vibration_g,
        current_a=payload.current_a,
        sample_rate_hz=payload.sample_rate_hz,
        rpm=payload.rpm,
        hf_bearing_min_hz=settings.hf_bearing_min_hz,
    )
    inference = run_tinyml_inference(
        features,
        temperature_c=payload.temperature_c,
        rpm=payload.rpm,
    )
    ts = utc_now_iso()

    try:
        repo.insert_recording(
            recording_id=payload.recording_id,
            machine_id=payload.machine_id,
            motor_id=payload.motor_id,
            timestamp=ts,
            sample_rate_hz=payload.sample_rate_hz,
            rpm=payload.rpm,
            load_pct=payload.load_pct,
            temperature_c=payload.temperature_c,
            label=payload.label,
            is_synthetic=False,
            vibration_g=payload.vibration_g,
            current_a=payload.current_a,
            features=features,
            inference=inference,
        )
    except sqlite3.IntegrityError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"recording_id '{payload.recording_id}' already exists",
        ) from exc

    snapshot = state_mgr.update_state(
        machine_id=payload.machine_id,
        inference=inference,
        rpm=payload.rpm,
        temperature_c=payload.temperature_c,
        updated_at=ts,
    )
    dashboard = state_mgr.build_dashboard_payload(snapshot, language=language)

    return RecordingIngestResponse(
        recording_id=payload.recording_id,
        machine_id=payload.machine_id,
        motor_id=payload.motor_id,
        timestamp=ts,
        features=features,
        inference=inference,
        dashboard=dashboard,
    )


@router.get(
    "/",
    response_model=RecordingListResponse,
    summary="List stored recordings with features and inference metadata",
)
def list_recordings(
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    machine_id: Optional[str] = Query(default=None),
    motor_id: Optional[str] = Query(default=None),
    label: Optional[str] = Query(default=None),
    predicted_fault: Optional[str] = Query(default=None),
    health_tier: Optional[int] = Query(default=None, ge=0, le=2),
    include_raw: bool = Query(default=True),
    repo: RecordingsRepository = Depends(get_repo),
) -> RecordingListResponse:
    """Retrieve stored recordings ordered newest-first with optional filters."""
    total, items = repo.list_recordings(
        limit=limit,
        offset=offset,
        machine_id=machine_id,
        motor_id=motor_id,
        label=label,
        predicted_fault=predicted_fault,
        health_tier=health_tier,
        include_raw=include_raw,
    )
    return RecordingListResponse(total=total, limit=limit, offset=offset, items=items)


@router.get(
    "/export/csv",
    summary="Export 28 extracted features and labels as CSV for TinyML model training",
)
def export_recordings_csv(
    machine_id: Optional[str] = Query(default=None),
    label: Optional[str] = Query(default=None),
    include_synthetic: bool = Query(default=True),
    repo: RecordingsRepository = Depends(get_repo),
) -> StreamingResponse:
    """Stream all matching recordings with their 28 features as a training-ready CSV."""
    rows = repo.iter_export_rows(
        machine_id=machine_id,
        label=label,
        include_synthetic=include_synthetic,
    )

    header = [
        "recording_id",
        "machine_id",
        "motor_id",
        "timestamp",
        "sample_rate_hz",
        "rpm",
        "load_pct",
        "temperature_c",
        "label",
        "is_synthetic",
        "sample_count",
        *FEATURE_NAMES,
        "health_tier",
        "health_status",
        "predicted_fault",
        "confidence",
        "anomaly_score",
    ]

    def _generate():
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(header)
        yield buf.getvalue()
        buf.seek(0)
        buf.truncate(0)

        for row in rows:
            writer.writerow(
                [
                    row["recording_id"],
                    row["machine_id"],
                    row["motor_id"],
                    row["timestamp"],
                    row["sample_rate_hz"],
                    row["rpm"],
                    row["load_pct"],
                    row["temperature_c"],
                    row["label"] or "",
                    row["is_synthetic"],
                    row["sample_count"],
                    *[row[name] for name in FEATURE_NAMES],
                    row["health_tier"],
                    row["health_status"],
                    row["predicted_fault"],
                    row["confidence"],
                    row["anomaly_score"],
                ]
            )
            yield buf.getvalue()
            buf.seek(0)
            buf.truncate(0)

    return StreamingResponse(
        _generate(),
        media_type="text/csv; charset=utf-8",
        headers={
            "Content-Disposition": 'attachment; filename="bldc_recordings_export.csv"'
        },
    )
