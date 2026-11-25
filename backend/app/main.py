"""FastAPI application entrypoint with lifespan hydration, CORS, and OpenAPI docs."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware

from app.api.routers import baseline, dashboard, inject, recordings
from app.core.config import Settings, get_settings
from app.core.state import DashboardStateManager
from app.db.repository import RecordingsRepository
from app.schemas import DashboardResponse


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure the FastAPI application instance."""
    cfg = settings or get_settings()

    @asynccontextmanager
    async def app_lifespan(app_instance: FastAPI):
        repo = RecordingsRepository(cfg.database_path)
        state_mgr = DashboardStateManager()
        state_mgr.hydrate(repo.get_latest_by_machine())
        app_instance.state.repo = repo
        app_instance.state.dashboard_state = state_mgr
        yield

    app = FastAPI(
        title=cfg.app_name,
        version=cfg.app_version,
        description=(
            "Production backend for Low-Cost BLDC Predictive Maintenance & Fault Detection "
            "using TinyML, designed for low-bandwidth mobile dashboards."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=app_lifespan,
    )

    if settings is not None:
        app.dependency_overrides[get_settings] = lambda: cfg

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.cors_origins,
        allow_credentials=cfg.cors_origins != ["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(recordings.router)
    app.include_router(inject.router)
    app.include_router(dashboard.router)
    app.include_router(baseline.router)

    @app.get("/health", tags=["System"])
    def healthcheck() -> dict[str, str]:
        return {"status": "ok", "version": cfg.app_version}

    @app.get(
        "/api/v1/status",
        response_model=DashboardResponse,
        tags=["Low-Bandwidth Dashboard"],
        summary="Get frontend-compatible machine status (vibration, current, sound, degradation)",
    )
    def get_latest_status(
        request: Request,
        machine_id: Optional[str] = Query(default=None),
        language: str = Query(default="en", pattern="^(en|hi)$"),
    ) -> DashboardResponse:
        mgr: DashboardStateManager = request.app.state.dashboard_state
        snap = mgr.get_snapshot(machine_id) if machine_id else mgr.get_latest_snapshot()
        if snap is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No telemetry recorded yet",
            )
        return mgr.build_dashboard_payload(snap, language=language)

    return app


app = create_app()
