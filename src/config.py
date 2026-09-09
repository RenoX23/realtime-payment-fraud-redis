"""Application configuration management using Pydantic Settings."""

from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralized application and infrastructure settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Redis Configuration
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: Optional[str] = None
    REDIS_SOCKET_TIMEOUT: float = 2.0

    # Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    MODEL_DIR: Path = BASE_DIR / "models"

    # Model Evaluation & Threshold Defaults
    TARGET_PR_AUC: float = 0.82
    DEFAULT_DECISION_THRESHOLD: float = 0.50
    MANUAL_REVIEW_LOWER_THRESHOLD: float = 0.30
    MANUAL_REVIEW_UPPER_THRESHOLD: float = 0.75

    # Financial Cost Assumptions (USD)
    COST_FALSE_NEGATIVE: float = 250.0  # Average chargeback loss + merchant penalty
    COST_FALSE_POSITIVE: float = 12.0   # Customer friction, support cost, lost checkout

    # API & Serving
    FASTAPI_HOST: str = "0.0.0.0"
    FASTAPI_PORT: int = 8000
    DEBUG: bool = False
    API_URL: str = "http://localhost:8000"


settings = Settings()
