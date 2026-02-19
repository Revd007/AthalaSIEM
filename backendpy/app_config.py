"""
AthalaSIEM Python backend configuration.
Reads from .env; same PostgreSQL as .NET backend.
"""
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings from environment."""

    # Database (same PostgreSQL as .NET backend)
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/athalasiem"

    # ML models directory (contains .pkl files)
    MODEL_DIR: str = "models"

    # Server
    PORT: int = 9797
    HOST: str = "0.0.0.0"

    # CORS (frontend on 7654, .NET backend on 9595)
    CORS_ORIGINS: list[str] = ["http://localhost:7654", "http://localhost:9595", "http://127.0.0.1:7654", "http://127.0.0.1:9595"]

    # Optional: Python backend URL for .NET proxy
    PYTHON_AI_BASE_URL: str = "http://localhost:9797"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "allow"


def get_model_dir() -> Path:
    """Resolve MODEL_DIR relative to backendpy root."""
    base = Path(__file__).resolve().parent
    return base / settings.MODEL_DIR


settings = Settings()
