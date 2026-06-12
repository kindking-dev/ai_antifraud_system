"""
SENTINEL AI: LLM Integration Schemas.
Defines strict API contracts for Explainable AI (XAI) interaction with LM Studio.
"""

from pydantic import BaseModel, Field, ConfigDict, field_validator
from typing import List, Dict, Literal


class LLMExplanationRequest(BaseModel):
    """
    Контракт входящего запроса на генерацию объяснения.
    Принимает результаты Late Fusion (CatBoost + HMOG) и SHAP-веса.
    """
    
    model_config = ConfigDict(extra="ignore", strict=True, str_strip_whitespace=True)

    transaction_id: str = Field(
        ..., 
        min_length=5, 
        max_length=100, 
        description="Уникальный ID транзакции для контекста"
    )
    fraud_probability: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Итоговый скор риска от CatBoost"
    )
    action: str = Field(
        ..., 
        description="Вердикт антифрод-системы (ALLOW, CHALLENGE, BLOCK)"
    )
    reason_codes: List[str] = Field(
        default_factory=list, 
        description="Список сработавших бизнес-правил (например, SUSPICIOUS_BEHAVIOR)"
    )
    feature_impacts: Dict[str, float] = Field(
        default_factory=dict, 
        description="Словарь SHAP-весов {feature_name: impact_value}"
    )

    @field_validator("transaction_id")
    @classmethod
    def validate_transaction_id(cls, v: str) -> str:
        """Очистка ID для предотвращения поломки системного промпта LLM"""
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Transaction ID cannot be empty")
        return cleaned


class LLMExplanationResponse(BaseModel):
    """
    Контракт исходящего ответа от LLM-сервиса обратно на фронтенд/дашборд.
    """
    
    model_config = ConfigDict(extra="ignore", strict=True)

    transaction_id: str = Field(
        ..., 
        description="ID транзакции (для маппинга ответа на клиенте)"
    )
    explanation_markdown: str = Field(
        ..., 
        description="Сгенерированный текст отчета от Qwen в формате Markdown"
    )
    processing_time_ms: float = Field(
        ..., 
        ge=0.0, 
        description="Время генерации ответа нейросетью (SLA для Cold Path)"
    )
    status: Literal["success", "error", "fallback"] = Field(
        default="success", 
        description="Статус работы AI-ассистента"
    )