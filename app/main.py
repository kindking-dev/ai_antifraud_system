"""
Main entry point for the AI Behavioral Anti-Fraud Service.
Orchestrates application lifespan and registers core API routes.
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path

import structlog
from catboost import CatBoostClassifier
from fastapi import FastAPI
from fastapi.responses import ORJSONResponse

from app.core.config import settings
from app.core.state import ml_models
from app.api.v1 import scoring

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("initializing_api", version=settings.VERSION)
    try:
        # Улучшенный резолв путей
        base_dir = Path(__file__).resolve().parents[1]
        model_path = base_dir / "ml_artifacts" / "core_scorer.cbm"

        if not model_path.exists():
            # На защите важно, чтобы модель БЫЛА, поэтому тут лучше кинуть ошибку
            logger.error("model_artifact_missing", path=str(model_path))
            raise FileNotFoundError(f"Model missing: {model_path}")

        logger.info("loading_model_into_ram", model_path=str(model_path))
        model = CatBoostClassifier()
        model.load_model(str(model_path))

        # Ключ теперь синхронизирован
        ml_models["core_scorer"] = model
        logger.info("ml_models_loaded_successfully")

        yield  # Сервер готов к работе

    except Exception as e:
        logger.exception("critical_startup_failure", error=str(e))
        # Не делаем yield здесь, чтобы FastAPI не стартовал с битой моделью
        raise SystemExit(1)
    finally:
        logger.info("shutting_down_api", action="clearing_ml_models")
        ml_models.clear()


# Initialize high-performance FastAPI instance
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url="/docs",
    redoc_url=None,
    lifespan=lifespan,
    default_response_class=ORJSONResponse,
)

# Register the scoring engine router
app.include_router(scoring.router, prefix=settings.API_V1_STR, tags=["Scoring Engine"])


@app.get("/health", tags=["System Observability"])
async def health_check() -> dict:
    """
    Liveness probe for orchestration tools (Docker/Kubernetes).
    Checks if the system is operational and ML models are active in RAM.
    """
    is_model_loaded = "core_scorer" in ml_models
    return {
        "status": "operational",
        "version": settings.VERSION,
        "models_loaded": is_model_loaded,
        "environment": os.getenv("ENV", "development"),
    }
