"""Shared pytest fixtures providing isolated SQLite storage and FastAPI TestClient."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app


@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "test_tinyml.db"


@pytest.fixture()
def test_settings(db_path: Path) -> Settings:
    return Settings(
        database_path=str(db_path),
        sse_heartbeat_seconds=1.0,
    )


@pytest.fixture()
def client(test_settings: Settings) -> Iterator[TestClient]:
    app = create_app(test_settings)
    with TestClient(app) as tc:
        yield tc
