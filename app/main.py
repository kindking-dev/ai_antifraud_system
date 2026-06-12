"""
AI-POWERED BEHAVIORAL ANTI-FRAUD SERVICE: Main Application Entrypoint.
Handles App Lifespan, Middleware, Redis Pooling, ML Warmup, and XAI Routing.
"""

import logging
from contextlib import asynccontextmanager
from typing import Dict, Any

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings, validate_settings
# Импортируем оба роутера: горячий путь (scoring) и холодный путь аналитики (explain)
from app.api.v1 import scoring, explain
from app.repositories.redis_store import RedisStore

# Импортируем ML-логику для проверки и прогрева
from app.ml.inference.transaction_inference import predict_transaction_model, MODEL_LOADED, feature_builder

# =========================
# LOGGING SETUP (FAANG Standard)
# =========================
logging.basicConfig(level=settings.LOG_LEVEL, format=settings.LOG_FORMAT)

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ]
)
logger = structlog.get_logger(__name__)


# =========================
# LIFESPAN (RESOURCE MANAGEMENT)
# =========================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Управляет жизненным циклом приложения:
    1. Валидирует конфиг.
    2. Проверяет готовность ML-артефактов.
    3. Инициализирует пул соединений Redis.
    4. Прогревает модель (Warmup) для устранения Cold Start.
    """
    logger.info("startup_begin", version=settings.VERSION)

    try:
        # 1. Валидация окружения
        validate_settings()

        # 2. Проверка ML-подсистемы
        if not MODEL_LOADED or not feature_builder.is_ready:
            logger.error("ml_system_offline", reason="CatBoost or Builder artifacts missing")
            raise RuntimeError("ML System failed to initialize. Check ml_artifacts folder.")
        
        logger.info("ml_system_ready", features_count=len(feature_builder.feature_columns))

        # 3. Инициализация Redis
        app.state.redis_store = RedisStore(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            socket_timeout=settings.REDIS_SOCKET_TIMEOUT
        )
        logger.info("redis_initialized", host=settings.REDIS_HOST)

        # 4. Интеллектуальный прогрев модели (Model Warmup)
        try:
            warmup_payload: Dict[str, Any] = {
                "transaction_id": "warmup_001",
                "user_id": "system_warmup",
                "amount_kzt": 1000.0,
                "timestamp_utc": "2026-04-30T12:00:00",
                "card1": "1234",
                "addr1": "100"
            }
            
            for _ in range(5):
                predict_transaction_model(warmup_payload, behavior_score=0.5)
                
            logger.info("model_warmed_up", status="success")
        except Exception as warmup_err:
            logger.warning("warmup_failed", error=str(warmup_err))

        logger.info("startup_complete", status="listening")
        yield

    except Exception as e:
        logger.exception("startup_failed", error=str(e))
        raise SystemExit(1)
    
    finally:
        logger.info("shutdown_begin")
        if hasattr(app.state, "redis_store"):
            await app.state.redis_store.close()
            
        logger.info("shutdown_complete")


# =========================
# FASTAPI APP INSTANCE
# =========================
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="AI-Powered Behavioral Anti-Fraud Service - Dual-Engine Fraud Detection System with Local LLM",
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url="/docs",
    lifespan=lifespan,
)

# =========================
# MIDDLEWARE
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# ROUTERS
# =========================
# 1. Hot Path: Основной скоринг (SLA < 50ms)
app.include_router(
    scoring.router,
    prefix=settings.API_V1_STR,
    tags=["Scoring Engine"],
)

# 2. Cold Path: Аналитика XAI через локальную нейросеть
app.include_router(
    explain.router,
    prefix=settings.API_V1_STR,
    # Теги уже определены внутри самого роутера
)


# =========================
# SYSTEM ENDPOINTS
# =========================
@app.get("/health", tags=["System"])
async def health_check():
    """
    Проверка работоспособности системы.
    Возвращает статус готовности ML и Redis.
    """
    return {
        "status": "operational",
        "version": settings.VERSION,
        "ml_engine_ready": MODEL_LOADED and feature_builder.is_ready,
        "redis_active": hasattr(app.state, "redis_store")
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)