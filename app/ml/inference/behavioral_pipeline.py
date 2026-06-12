"""
SENTINEL AI: Behavioral Biometrics Inference Pipeline.
Optimized for high-concurrency and sub-10ms inference.
"""

import structlog
import json
from typing import Dict, List, Optional, Any
from pathlib import Path
from catboost import CatBoostClassifier

from app.core.config import settings
from app.repositories.redis_store import RedisStore

logger = structlog.get_logger(__name__)

# Пути к артефактам
BASE_DIR = Path(__file__).resolve().parents[3]
MODEL_PATH = BASE_DIR / "ml_artifacts" / "behavioral_similarity.cbm"
SCHEMA_PATH = BASE_DIR / "ml_artifacts" / "behavioral_schema.json"


class BehavioralPipeline:
    def __init__(self):
        self.model = CatBoostClassifier()
        self._load_model()
        self._load_feature_schema()
        # Используем существующий репозиторий для доступа к пулу соединений
        self.redis_store = RedisStore()

    def _load_model(self) -> None:
        if MODEL_PATH.exists():
            try:
                self.model.load_model(str(MODEL_PATH))
                logger.info("behavioral_model_loaded", path=str(MODEL_PATH))
            except Exception as e:
                logger.error("behavioral_model_load_failed", error=str(e))
        else:
            logger.error("behavioral_model_not_found", path=str(MODEL_PATH))

    def _load_feature_schema(self) -> None:
        """Загружает порядок фичей, чтобы не нарушить feature contract модели."""
        if SCHEMA_PATH.exists():
            try:
                with open(SCHEMA_PATH, "r") as f:
                    schema = json.load(f)
                    self.feature_names = schema.get("features", [])
                logger.info("behavioral_schema_loaded", count=len(self.feature_names))
            except Exception as e:
                logger.error("behavioral_schema_load_error", error=str(e))
                self._set_default_features()
        else:
            self._set_default_features()

    def _set_default_features(self) -> None:
        self.feature_names = [
            "duration_ms_mean",
            "duration_ms_std",
            "duration_ms_max",
            "length_px_mean",
            "length_px_std",
            "length_px_max",
            "velocity_mean",
            "velocity_std",
            "velocity_max",
            "median_pressure_mean",
            "median_pressure_std",
            "median_pressure_max",
            "median_area_mean",
            "median_area_std",
            "median_area_max",
        ]

    async def predict_fraud_score(
        self, user_id: str, current_features: Dict[str, Any]
    ) -> float:
        """
        Выполняет инференс поведенческой модели.
        1. Получает профиль (Digital Twin) из Redis.
        2. Считает дельты между текущей сессией и эталоном.
        3. Выполняет скоринг CatBoost.
        4. Обновляет состояние в Redis для Late Fusion.
        """
        try:
            profile_key = f"user:{user_id}:profile"

            # 1. Fetch profile (используем клиент из пула RedisStore)
            profile_data = await self.redis_store.client.hgetall(profile_key)

            if not profile_data:
                logger.warning("behavioral_profile_not_found", user_id=user_id)
                return 0.5  # Нейтральный скор (Cold Start)

            # 2. Build input vector (Delta calculation)
            # Модель обучалась на разности между поведением владельца и текущим вводом
            input_vector: List[float] = []

            for col in self.feature_names:
                # Пытаемся найти значение с префиксом и без (защита от ошибок загрузки)
                prof_raw = profile_data.get(f"prof_{col}") or profile_data.get(col)
                prof_val = float(prof_raw) if prof_raw else 0.0

                win_val = float(current_features.get(col, 0.0))

                # Вычисляем абсолютную дельту (нормализация может быть внутри модели)
                input_vector.append(abs(win_val - prof_val))

            # 3. Model Inference (уходим от Pandas к List[List] для скорости)
            # CatBoostClassifier.predict_proba возвращает [[prob_0, prob_1]]
            # target 0 обычно Fraud, target 1 обычно Legit в задачах бинарной классификации
            pred_proba = self.model.predict_proba([input_vector])
            fraud_prob = float(pred_proba[0][0])  # Вероятность класса 0

            # 4. Update Late Fusion State (Async)
            state_key = f"user:{user_id}:state"
            await self.redis_store.client.hset(
                state_key, "latest_behavior_score", fraud_prob
            )

            # Устанавливаем TTL для состояния (например, 30 минут), чтобы не забивать Redis
            await self.redis_store.client.expire(state_key, 1800)

            return round(fraud_prob, 4)

        except Exception as e:
            logger.exception(
                "behavioral_inference_critical_error", user_id=user_id, error=str(e)
            )
            return 0.5  # Fail-safe нейтральный скор
