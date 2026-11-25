"""FastAPI application entrypoint with lifespan hydration, CORS, and OpenAPI docs."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routers import baseline, dashboard, inject, recordings
from app.core.config import Settings, get_settings
from app.core.state import DashboardStateManager
from app.db.repository import RecordingsRepository


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

    return app


app = create_app()
