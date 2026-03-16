from typing import Optional, List
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict, IPvAnyAddress


class TransactionSource(str, Enum):
    """Каналы поступления транзакций."""

    MOBILE_APP = "mobile_app"
    WEB_BROWSER = "web_browser"
    API_B2B = "api_b2b"


class BiometricSensors(BaseModel):
    """
    Инновация 1: Sensor Fusion & Behavioral Biometrics.
    Телеметрия с датчиков телефона для защиты от RAT (Remote Access Trojans) и эмуляторов.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    gyroscope_x_y_z: List[float] = Field(
        ...,
        min_length=3,
        max_length=3,
        description="Вектор наклона устройства [X, Y, Z]",
    )
    keystroke_entropy: float = Field(
        ...,
        ge=0.0,
        description="Энтропия ритма печати (защита от скриптов автозаполнения)",
    )
    touch_pressure_variance: Optional[float] = Field(
        None,
        description="Дисперсия силы нажатия на экран (опционально, зависит от экрана)",
    )


class NetworkIdentity(BaseModel):
    """
    Инновация 2: Network-Layer Fingerprinting.
    Криптографическая проверка сетевого пакета.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    ip_address: IPvAnyAddress
    ja3_fingerprint: str = Field(
        ...,
        min_length=32,
        max_length=32,
        description="MD5 хеш TLS-рукопожатия. Отсеивает ботнеты.",
    )
    user_agent: str
    is_vpn_or_proxy: bool = Field(default=False)


class FraudAnalysisRequest(BaseModel):
    """
    Главный контракт для эндпоинта /v1/score-transaction.
    Написан на Rust (под капотом Pydantic V2) для валидации за <1ms.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    transaction_id: str = Field(..., min_length=5, max_length=50)
    user_id: str = Field(..., description="Уникальный ID клиента в системе банка")
    amount_kzt: float = Field(..., gt=0.0, description="Сумма транзакции в тенге")

    # Метаданные транзакции
    source: TransactionSource
    network: NetworkIdentity
    biometrics: Optional[BiometricSensors] = None

    # Инновация 3: Continuous Authentication
    session_trust_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Индекс доверия к текущей сессии (падает при аномалиях)",
    )
    timestamp_utc: datetime = Field(default_factory=datetime.utcnow)
