"""
API Response Schemas for Fraud Analysis.
Ensures strict validation of output data using Pydantic V2.
Updated to support SHAP Feature Impact visualization.
"""

from typing import List, Dict
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict


class FraudAction(str, Enum):
    ALLOW = "ALLOW"
    CHALLENGE = "CHALLENGE"
    BLOCK = "BLOCK"


class ReasonCode(str, Enum):
    """
    Standardized Reason Codes for AML and Internal Audits.
    Names MUST strictly match usage in the scoring logic.
    """

    HIGH_ML_RISK = "HIGH_ML_RISK"
    SUSPICIOUS_BEHAVIOR = "SUSPICIOUS_BEHAVIOR"
    AML_VELOCITY_SPIKE = "AML_VELOCITY_SPIKE"
    DEVICE_ANOMALY = "RAT_DEVICE_ANOMALY"
    NETWORK_RISK = "SUSPICIOUS_NETWORK_LINK"
    JA3_MISMATCH = "API_SPOOFING_DETECTED"


class FraudAnalysisResponse(BaseModel):
    """
    Основная схема ответа антифрод-движка.
    Включает XAI-компоненты для обеспечения прозрачности (Explainability).
    """

    model_config = ConfigDict(strict=True)

    transaction_id: str = Field(..., description="Unique transaction reference")
    action: FraudAction = Field(..., description="Final decision from ML ensemble")
    fraud_probability: float = Field(
        ..., ge=0.0, le=1.0, description="Probability score [0-1]"
    )
    reason_codes: List[ReasonCode] = Field(
        default_factory=list, description="List of SHAP-derived triggers"
    )

    # НОВОЕ ПОЛЕ ДЛЯ ЗАДАЧИ №5
    feature_impacts: Dict[str, float] = Field(
        default_factory=dict,
        description="SHAP weights for each input feature for visualization",
    )

    processing_time_ms: float = Field(
        ..., description="Total inference latency (SLA < 50ms)"
    )
