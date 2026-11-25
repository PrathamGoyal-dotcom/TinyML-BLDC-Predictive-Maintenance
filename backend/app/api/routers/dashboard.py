"""Low-Bandwidth Dashboard Serving endpoints (/api/v1/dashboard/{machine_id})."""

from __future__ import annotations

import asyncio
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import StreamingResponse

from app.core.config import Settings, get_settings
from app.core.state import DashboardStateManager
from app.schemas import DashboardResponse

router = APIRouter(prefix="/api/v1/dashboard", tags=["Low-Bandwidth Dashboard"])


def get_state_manager(request: Request) -> DashboardStateManager:
    return request.app.state.dashboard_state


@router.get(
    "/",
    response_model=list[DashboardResponse],
    summary="List compact localized health status for all active machines",
)
def list_machine_dashboards(
    language: str = Query(default="en", pattern="^(en|hi)$"),
    state_mgr: DashboardStateManager = Depends(get_state_manager),
) -> list[DashboardResponse]:
    """Return compact dashboard payloads for all machines with live state."""
    responses: list[DashboardResponse] = []
    for machine_id in state_mgr.list_machine_ids():
        snap = state_mgr.get_snapshot(machine_id)
        if snap is not None:
            responses.append(state_mgr.build_dashboard_payload(snap, language=language))
    return responses


@router.get(
    "/{machine_id}",
    response_model=DashboardResponse,
    summary="Get ultra-lightweight localized health status for a machine",
)
def get_machine_dashboard(
    machine_id: str,
    language: str = Query(default="en", pattern="^(en|hi)$"),
    state_mgr: DashboardStateManager = Depends(get_state_manager),
) -> DashboardResponse:
    """Return compact machine health tier, fault type, and localized farmer/workshop hints."""
    snapshot = state_mgr.get_snapshot(machine_id)
    if snapshot is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No telemetry recorded yet for machine '{machine_id}'",
        )
    return state_mgr.build_dashboard_payload(snapshot, language=language)


@router.get(
    "/{machine_id}/stream",
    summary="Server-Sent Events (SSE) stream of compact live machine health updates",
)
async def stream_machine_dashboard(
    machine_id: str,
    request: Request,
    language: str = Query(default="en", pattern="^(en|hi)$"),
    max_events: Optional[int] = Query(
        default=None,
        ge=1,
        le=10000,
        description="Optional cap on SSE data events before closing (useful for testing or polling).",
    ),
    heartbeat_s: Optional[float] = Query(
        default=None,
        gt=0.1,
        le=120.0,
        description="Optional keepalive heartbeat interval in seconds.",
    ),
    state_mgr: DashboardStateManager = Depends(get_state_manager),
    settings: Settings = Depends(get_settings),
) -> StreamingResponse:
    """Stream ultra-lightweight SSE health updates whenever telemetry arrives or faults are injected."""
    loop = asyncio.get_running_loop()
    queue = state_mgr.subscribe(machine_id, loop)
    keepalive = heartbeat_s if heartbeat_s is not None else settings.sse_heartbeat_seconds

    async def _event_stream():
        sent_events = 0
        try:
            initial = state_mgr.get_snapshot(machine_id)
            if initial is not None:
                payload = state_mgr.build_dashboard_payload(initial, language=language)
                sent_events += 1
                yield f"id: {sent_events}\nevent: status\ndata: {payload.model_dump_json()}\n\n"
                if max_events is not None and sent_events >= max_events:
                    return
            else:
                yield f": connected {machine_id}\n\n"

            while True:
                if await request.is_disconnected():
                    break
                try:
                    snapshot = await asyncio.wait_for(queue.get(), timeout=keepalive)
                except asyncio.TimeoutError:
                    yield ": ping\n\n"
                    continue

                payload = state_mgr.build_dashboard_payload(snapshot, language=language)
                sent_events += 1
                yield f"id: {sent_events}\nevent: status\ndata: {payload.model_dump_json()}\n\n"
                if max_events is not None and sent_events >= max_events:
                    break
        finally:
            state_mgr.unsubscribe(machine_id, queue)

    return StreamingResponse(
        _event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
