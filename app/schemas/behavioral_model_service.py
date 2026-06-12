"""
SENTINEL AI: Behavioral Biometrics Schemas.
Defines API contracts for raw telemetry processing and risk scoring.
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone

class BehavioralWindowFeatures(BaseModel):
    """
    Статистические признаки окна из 15 свайпов (Touchalytics features).
    """
    model_config = ConfigDict(populate_by_name=True)

    # Duration features
    duration_ms_mean: float = Field(..., ge=0, description="Средняя длительность свайпа")
    duration_ms_std: float = Field(..., ge=0, description="Стандартное отклонение длительности")
    duration_ms_max: float = Field(..., ge=0, description="Максимальная длительность свайпа")

    # Length features
    length_px_mean: float = Field(..., ge=0, description="Средняя длина траектории в пикселях")
    length_px_std: float = Field(..., ge=0, description="Стандартное отклонение длины")
    length_px_max: float = Field(..., ge=0, description="Максимальная длина свайпа")

    # Velocity features
    velocity_mean: float = Field(..., ge=0, description="Средняя скорость движения")
    velocity_std: float = Field(..., ge=0, description="Стандартное отклонение скорости")
    velocity_max: float = Field(..., ge=0, description="Максимальная скорость")

    # Pressure features
    median_pressure_mean: float = Field(..., ge=0, description="Среднее медианное давление")
    median_pressure_std: float = Field(..., ge=0, description="Стандартное отклонение давления")
    median_pressure_max: float = Field(..., ge=0, description="Максимальное давление")

    # Area features
    median_area_mean: float = Field(..., ge=0, description="Средняя площадь касания")
    median_area_std: float = Field(..., ge=0, description="Стандартное отклонение площади")
    median_area_max: float = Field(..., ge=0, description="Максимальная площадь касания")


class BehavioralInferenceRequest(BaseModel):
    """Запрос на проверку поведенческой биометрии."""
    user_id: str = Field(..., description="ID пользователя")
    device_id: Optional[str] = None
    
    # 🔥 НОВОЕ: Принимаем массив сырых событий (координаты, время, давление)
    events: Optional[List[Dict[str, Any]]] = Field(
        default=None, 
        description="Сырые события свайпов (x, y, timestamp_ms, pressure, contact_size)"
    )
    
    # СТАРОЕ: Сделали опциональным, чтобы не ломать старые запросы
    features: Optional[BehavioralWindowFeatures] = Field(
        default=None, 
        description="Агрегированные признаки (устарело, используйте events)"
    )
    
    # Используем timezone-aware datetime для production
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class BehavioralInferenceResponse(BaseModel):
    """Результат оценки поведенческого риска."""
    user_id: str
    fraud_probability: float = Field(..., ge=0, le=1.0, description="Вероятность того, что это мошенник")
    is_anomaly: bool = Field(..., description="Флаг аномалии на основе порога")
    processing_time_ms: float
    status: str = Field(..., description="MATCH / IMPOSTOR / COLD_START / ERROR")


class UserProfileState(BaseModel):
    """Стейт пользователя в Redis для Late Fusion."""
    latest_behavior_score: float
    last_updated: datetime
    is_active_session: bool