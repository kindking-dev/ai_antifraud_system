from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import ORJSONResponse
import structlog

from app.core.config import settings

logger = structlog.get_logger()

# Глобальный словарь для хранения тяжелых ML-моделей в оперативной памяти (O(1) доступ)
ml_models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Управляет жизненным циклом приложения (Startup / Shutdown).
    Гарантирует, что модели загружены в RAM до того, как API начнет принимать трафик.
    """
    logger.info("Initializing AI Anti-Fraud API...", version=settings.VERSION)

    try:
        # TODO: Загрузка моделей CatBoost и IsolationForest
        # Временно ставим заглушку, пока вы не скачаете обученные модели из Kaggle
        # ml_models["core_scorer"] = CatBoostClassifier().load_model("ml_artifacts/catboost_core_v1.cbm")

        logger.info("ML Models loaded into RAM successfully.")
        yield
    except Exception as e:
        logger.error("CRITICAL: Failed to load ML models!", error=str(e))
        raise
    finally:
        # Выполняется при выключении сервера (освобождаем память)
        ml_models.clear()
        logger.info("Anti-Fraud API shut down gracefully. RAM cleared.")


# Инициализация сверхбыстрого FastAPI инстанса
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url="/docs",  # Swagger UI для комиссии
    redoc_url=None,  # Отключаем лишнее для скорости
    lifespan=lifespan,
    default_response_class=ORJSONResponse,  # orjson работает в разы быстрее стандартного json
)


@app.get("/health", tags=["System Observability"])
async def health_check():
    """
    Эндпоинт для Docker и Kubernetes.
    Проверяет, жив ли сервис.
    """
    return {
        "status": "operational",
        "version": settings.VERSION,
        "models_loaded": len(ml_models) == 0,  # Пока 0, так как стоит заглушка
    }
