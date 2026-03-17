from typing import Optional, List
from datetime import datetime, timezone
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict, IPvAnyAddress


class TransactionSource(str, Enum):
    MOBILE_APP = "MOBILE_APP"
    WEB = "WEB"
    API = "API"


class BiometricSensors(BaseModel):
    """Инновация 1: Sensor Fusion & Behavioral Biometrics."""

    model_config = ConfigDict(extra="forbid")

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
        None,
        description="Дисперсия силы нажатия на экран",
    )


class NetworkIdentity(BaseModel):
    """Инновация 2: Network-Layer Fingerprinting."""

    model_config = ConfigDict(extra="forbid")

    ip_address: IPvAnyAddress
    ja3_fingerprint: str = Field(
        ...,
        min_length=32,
        max_length=32,
        description="MD5 хеш TLS-рукопожатия.",
    )
    user_agent: str
    is_vpn_or_proxy: bool = Field(default=False)


class FraudAnalysisRequest(BaseModel):
    """Главный контракт для эндпоинта /v1/score-transaction."""

    model_config = ConfigDict(extra="forbid")

    transaction_id: str = Field(..., min_length=5, max_length=50)
    user_id: str = Field(..., description="Уникальный ID клиента в системе банка")
    amount_kzt: float = Field(..., gt=0.0, description="Сумма транзакции в тенге")

    source: TransactionSource
    network: NetworkIdentity
    biometrics: Optional[BiometricSensors] = None

    session_trust_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Индекс доверия к текущей сессии",
    )
    # Используем timezone.utc вместо deprecated datetime.utcnow()
    timestamp_utc: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
