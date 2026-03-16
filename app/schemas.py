from typing import Optional, List
from pydantic import BaseModel, Field, ConfigDict, IPvAnyAddress
from datetime import datetime
from enum import Enum


class TransactionSource(str, Enum):
    MOBILE_APP = "mobile_app"
    WEB_BROWSER = "web_browser"
    API_B2B = "api_b2b"


class BiometricSensors(BaseModel):
    """Real-time hardware sensor telemetry for Zero-Day RAT detection."""

    model_config = ConfigDict(extra="forbid", strict=True)

    gyroscope_x_y_z: List[float] = Field(
        ..., min_length=3, max_length=3, description="Device tilt vector"
    )
    keystroke_entropy: float = Field(
        ..., ge=0.0, description="Dwell/Flight time entropy"
    )
    touch_pressure_variance: Optional[float] = Field(
        None, description="Screen pressure standard deviation"
    )


class NetworkIdentity(BaseModel):
    """Network-layer fingerprinting to prevent API spoofing."""

    model_config = ConfigDict(extra="forbid", strict=True)

    ip_address: IPvAnyAddress
    ja3_fingerprint: str = Field(
        ..., min_length=32, max_length=32, description="MD5 hash of TLS handshake"
    )
    user_agent: str
    is_vpn_or_proxy: bool = Field(default=False)


class FraudAnalysisRequest(BaseModel):
    """
    Primary API contract for the /v1/score-transaction endpoint.
    Optimized for Rust-based Pydantic V2 validation (<1ms).
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    transaction_id: str = Field(..., min_length=5, max_length=50)
    user_id: str = Field(..., description="Unique customer identifier")
    amount_kzt: float = Field(..., gt=0.0)

    # Context
    source: TransactionSource
    network: NetworkIdentity
    biometrics: Optional[BiometricSensors] = None

    # Internal session state
    session_trust_score: float = Field(
        ..., ge=0.0, le=1.0, description="Exponentially decaying trust index"
    )
    timestamp_utc: datetime = Field(default_factory=datetime.utcnow)
