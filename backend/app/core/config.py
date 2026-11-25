"""Application configuration loaded from environment variables."""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings for the TinyML Predictive Maintenance backend."""

    app_name: str = "TinyML Predictive Maintenance API"
    app_version: str = "1.0.0"
    database_path: str = "./data/tinyml.db"
    cors_origins: list[str] = ["*"]
    sse_heartbeat_seconds: float = 15.0
    default_language: str = "en"
    hf_bearing_min_hz: float = 250.0

    model_config = SettingsConfigDict(
        env_prefix="TINYML_",
        env_file=".env",
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached application settings."""
    return Settings()
