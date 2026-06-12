import warnings
from pathlib import Path
from typing import Optional

from pydantic import Field, computed_field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic_settings")


class Settings(BaseSettings):
    """
    Production-grade configuration for antifraud system.

    Guarantees:
    - strict typing
    - SLA-aware config
    - ML + Redis alignment
    - secure defaults
    """

    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).resolve().parents[2] / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # =========================
    # APP
    # =========================
    PROJECT_NAME: str = "AI Anti-Fraud System"
    VERSION: str = "2.0.0"
    ENV: str = "dev"  # dev / staging / production
    DEBUG: bool = False

    API_V1_STR: str = "/api/v1"

    # =========================
    # SECURITY
    # =========================
    API_KEY: str = "DEV-KEY-CHANGE-ME"
    SECRET_KEY: Optional[str] = None

    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # =========================
    # POSTGRES
    # =========================
    POSTGRES_USER: str = "admin"
    POSTGRES_PASSWORD: str = Field(
        default="secretpassword",
        json_schema_extra={"writeOnly": True},
    )
    POSTGRES_HOST: str = "127.0.0.1"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "antifraud_db"

    DB_POOL_MIN_SIZE: int = 1
    DB_POOL_MAX_SIZE: int = 10
    DB_TIMEOUT: float = 5.0

    @computed_field(return_type=str)
    def SQLALCHEMY_DATABASE_URI(self) -> str:
        return (
            f"postgresql+asyncpg://"
            f"{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@"
            f"{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/"
            f"{self.POSTGRES_DB}"
        )

    # =========================
    # REDIS (CRITICAL FOR BEHAVIOR)
    # =========================
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: Optional[str] = None

    # SLA-critical settings
    REDIS_SOCKET_TIMEOUT: float = 0.03  # 30ms max
    REDIS_MAX_CONNECTIONS: int = 20

    # Behavioral windows (MUST match engine)
    WINDOW_1MIN: int = 60
    WINDOW_5MIN: int = 300

    @computed_field(return_type=str)
    def REDIS_URI(self) -> str:
        if self.REDIS_PASSWORD:
            return (
                f"redis://:{self.REDIS_PASSWORD}@"
                f"{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
            )
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"

    # =========================
    # ML CONFIG
    # =========================
    MODEL_PATH: str = "ml_artifacts/behavioral_model.cbm"
    TRANSACTION_MODEL_PATH: str = "ml_artifacts/catboost_final.cbm"

    ARTIFACTS_DIR: str = "ml_artifacts"

    # HYBRID MODEL WEIGHTS
    TX_WEIGHT: float = 0.7
    BEHAVIOR_WEIGHT: float = 0.3

    # VALIDATE WEIGHTS
    @field_validator("BEHAVIOR_WEIGHT")
    def validate_weights(cls, v, values):
        tx = values.data.get("TX_WEIGHT", 0.7)
        if abs((tx + v) - 1.0) > 0.01:
            raise ValueError("TX_WEIGHT + BEHAVIOR_WEIGHT must sum to 1")
        return v

    # =========================
    # FRAUD THRESHOLDS
    # =========================
    FRAUD_THRESHOLD_BLOCK: float = 0.85
    FRAUD_THRESHOLD_CHALLENGE: float = 0.5

    # =========================
    # SLA / PERFORMANCE
    # =========================
    SLA_LATENCY_LIMIT_MS: float = 50.0
    MAX_ALLOWED_LATENCY_MS: float = 50.0

    # Timeouts for safety
    MODEL_TIMEOUT_MS: float = 20.0
    BEHAVIOR_TIMEOUT_MS: float = 20.0

    # =========================
    # LOGGING
    # =========================
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    ENABLE_TELEMETRY_CONSOLE_LOGS: bool = False

    # =========================
    # VALIDATION
    # =========================
    def validate_production_ready(self) -> list[str]:
        issues = []

        if self.ENV == "production":
            if self.DEBUG:
                issues.append("DEBUG must be False in production")

            if "DEV" in self.API_KEY:
                issues.append("API_KEY must be secure")

            if self.POSTGRES_PASSWORD == "secretpassword":
                issues.append("POSTGRES_PASSWORD is weak")

            if not self.SECRET_KEY:
                issues.append("SECRET_KEY required in production")

            if self.REDIS_SOCKET_TIMEOUT > 0.05:
                issues.append("Redis timeout too high for SLA")

        return issues

    def to_safe_dict(self) -> dict:
        return {
            "env": self.ENV,
            "version": self.VERSION,
            "redis_host": self.REDIS_HOST,
            "postgres_host": self.POSTGRES_HOST,
            "sla_ms": self.SLA_LATENCY_LIMIT_MS,
        }


# =========================
# SINGLETON
# =========================

settings = Settings()


# =========================
# STARTUP VALIDATION
# =========================

def validate_settings():
    issues = settings.validate_production_ready()

    if issues:
        msg = "\n".join(f"⚠️ {i}" for i in issues)

        if settings.ENV == "production":
            raise ValueError(f"Invalid production config:\n{msg}")

        import logging
        logger = logging.getLogger(__name__)
        for i in issues:
            logger.warning(f"Config warning: {i}")  
