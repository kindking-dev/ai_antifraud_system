from typing import Optional, List, Dict
from datetime import datetime, timezone
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict, IPvAnyAddress, field_validator


class TransactionSource(str, Enum):
    MOBILE_APP = "MOBILE_APP"
    WEB = "WEB"
    API = "API"


class BiometricSensors(BaseModel):
    """
    Инновация 1: Sensor Fusion & Behavioral Biometrics.
    Игнорируем лишнее, чтобы не ломать инференс при обновлении сенсоров на клиенте.
    """
    model_config = ConfigDict(
        extra="ignore",
        populate_by_name=True,
        validate_assignment=True
    )

    gyroscope_x_y_z: List[float] = Field(
        ...,
        min_length=3,
        max_length=3,
        description="Вектор наклона устройства [X, Y, Z]",
    )
    keystroke_entropy: float = Field(
        ...,
        ge=0.0,
        description="Энтропия ритма печати",
    )
    touch_pressure_variance: Optional[float] = Field(
        0.0,
        description="Дисперсия силы нажатия на экран. Дефолт 0.0 для стабильности ML.",
    )


class NetworkIdentity(BaseModel):
    """
    Инновация 2: Network-Layer Fingerprinting.
    """
    model_config = ConfigDict(extra="ignore")

    ip_address: IPvAnyAddress
    ja3_fingerprint: str = Field(
        ...,
        description="MD5 хеш TLS-рукопожатия (32 символа).",
    )
    user_agent: str = Field(..., min_length=1)
    is_vpn_or_proxy: bool = Field(default=False)

    @field_validator("ja3_fingerprint")
    @classmethod
    def validate_ja3(cls, v: str) -> str:
        v = v.lower().strip()
        if len(v) != 32:
            raise ValueError("JA3 fingerprint must be exactly 32 chars (MD5)")
        return v


class FraudAnalysisRequest(BaseModel):
    """
    Главный контракт для эндпоинта /v1/score-transaction.
    РЕЖИМ 'ИМБА': Принимаем всё, используем только нужное, никогда не падаем.
    """
    model_config = ConfigDict(
        extra="ignore",  # Игнорируем лишние поля фронтенда
        str_strip_whitespace=True, # Чистим пробелы в ID
        use_enum_values=True # Для совместимости с CatBoost
    )

    transaction_id: str = Field(..., min_length=5, max_length=100)
    user_id: str = Field(..., description="Уникальный ID клиента в системе банка")
    amount_kzt: float = Field(..., gt=0.0, description="Сумма транзакции в тенге")

    source: TransactionSource
    network: NetworkIdentity
    biometrics: Optional[BiometricSensors] = None

    session_trust_score: float = 0.5
    
    @field_validator("session_trust_score")
    @classmethod
    def validate_trust_score(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError("session_trust_score must be between 0.0 and 1.0")
        return v
    
    timestamp_utc: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Время транзакции по UTC"
    )

    @field_validator("user_id")
    @classmethod
    def normalize_user_id(cls, v: str) -> str:
        return v.strip().lower()


class FraudAnalysisResponse(BaseModel):
    """Стандартный ответ системы."""
    transaction_id: str
    action: str = Field(..., description="ALLOW / CHALLENGE / BLOCK")
    fraud_probability: float = Field(..., ge=0.0, le=1.0)
    reason_codes: List[str] = []
    
    # 🔥 ИСПРАВЛЕНИЕ: Добавлено поле для передачи SHAP весов в XAI и на Дашборд
    feature_impacts: Dict[str, float]
    
    processing_time_ms: float