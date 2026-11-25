"""Failure Injection Pipeline endpoints (/api/v1/inject/)."""

from __future__ import annotations

import sqlite3
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from app.core.config import Settings, get_settings
from app.core.state import DashboardStateManager, utc_now_iso
from app.db.repository import RecordingsRepository
from app.ml.fault_injector import inject_fault
from app.ml.features import extract_28_features
from app.ml.inference import run_tinyml_inference
from app.schemas import InjectRequest, InjectResponse

router = APIRouter(prefix="/api/v1/inject", tags=["Failure Injection Pipeline"])


def get_repo(request: Request) -> RecordingsRepository:
    return request.app.state.repo


def get_state_manager(request: Request) -> DashboardStateManager:
    return request.app.state.dashboard_state


@router.post(
    "/",
    response_model=InjectResponse,
    status_code=status.HTTP_200_OK,
    summary="Inject controlled BLDC fault onto synthesized or user-supplied waveform",
)
def inject_bldc_fault(
    payload: InjectRequest,
    language: str = Query(default="en", pattern="^(en|hi)$"),
    repo: RecordingsRepository = Depends(get_repo),
    state_mgr: DashboardStateManager = Depends(get_state_manager),
    settings: Settings = Depends(get_settings),
) -> InjectResponse:
    """Inject a controlled fault, run TinyML inference, update live state, and optionally persist."""
    recording_id = payload.recording_id or f"inj-{uuid4()}"
    vibration_g, current_a, temperature_c = inject_fault(
        fault_type=payload.fault_type,
        severity=payload.severity,
        snr_db=payload.snr_db,
        drift_offset=payload.drift_offset,
        sample_rate_hz=payload.sample_rate_hz,
        rpm=payload.rpm,
        load_pct=payload.load_pct,
        temperature_c=payload.temperature_c,
        window_size=payload.window_size,
        base_vibration_g=payload.base_vibration_g,
        base_current_a=payload.base_current_a,
        seed=payload.seed,
    )

    features = extract_28_features(
        vibration_g=vibration_g,
        current_a=current_a,
        sample_rate_hz=payload.sample_rate_hz,
        rpm=payload.rpm,
        hf_bearing_min_hz=settings.hf_bearing_min_hz,
    )
    inference = run_tinyml_inference(
        features,
        temperature_c=temperature_c,
        rpm=payload.rpm,
    )
    ts = utc_now_iso()

    if payload.persist:
        try:
            repo.insert_recording(
                recording_id=recording_id,
                machine_id=payload.machine_id,
                motor_id=payload.motor_id,
                timestamp=ts,
                sample_rate_hz=payload.sample_rate_hz,
                rpm=payload.rpm,
                load_pct=payload.load_pct,
                temperature_c=temperature_c,
                label=payload.fault_type,
                is_synthetic=True,
                vibration_g=vibration_g,
                current_a=current_a,
                features=features,
                inference=inference,
            )
        except sqlite3.IntegrityError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"recording_id '{recording_id}' already exists",
            ) from exc

    snapshot = state_mgr.update_state(
        machine_id=payload.machine_id,
        inference=inference,
        rpm=payload.rpm,
        temperature_c=temperature_c,
        vib_std=features["vib_std"],
        curr_rms=features["curr_rms"],
        sound_db=payload.sound_db,
        updated_at=ts,
    )
    dashboard = state_mgr.build_dashboard_payload(snapshot, language=language)

    return InjectResponse(
        recording_id=recording_id,
        machine_id=payload.machine_id,
        motor_id=payload.motor_id,
        persisted=payload.persist,
        injected_fault=payload.fault_type,
        severity=payload.severity,
        snr_db=payload.snr_db,
        drift_offset=payload.drift_offset,
        sample_rate_hz=payload.sample_rate_hz,
        rpm=payload.rpm,
        temperature_c=temperature_c,
        features=features,
        inference=inference,
        dashboard=dashboard,
        vibration_g=vibration_g,
        current_a=current_a,
    )
