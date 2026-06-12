"""
Database ORM Models using SQLAlchemy 2.0.
Defines the schema for the Audit Trail.
"""

from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from sqlalchemy import String, Float, DateTime, JSON
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy ORM models."""
    pass


class TransactionLog(Base):
    """
    Аудит-лог транзакций. Обязателен для compliance в финтехе.
    Реализует хранение результатов Late Fusion скоринга.
    """

    __tablename__ = "transaction_logs"

    # Идентификатор транзакции
    transaction_id: Mapped[str] = mapped_column(String(50), primary_key=True)

    # Индекс для O(log N) поиска истории конкретного юзера
    user_id: Mapped[str] = mapped_column(String(50), index=True)

    # Финансовые данные
    amount_kzt: Mapped[float] = mapped_column(Float)
    
    # Результаты ML-моделей (Late Fusion)
    fraud_probability: Mapped[float] = mapped_column(Float)
    
    # Скор от первой модели (биометрия)
    behavior_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # Вердикт системы (ALLOW, CHALLENGE, BLOCK)
    action: Mapped[str] = mapped_column(String(20))

    # JSON-данные для объяснимости (Explainable AI)
    reason_codes: Mapped[Optional[List[str]]] = mapped_column(JSON, nullable=True)
    
    # Словарь SHAP-весов для дашборда
    feature_impacts: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)

    # Техническая метрика SLA
    processing_time_ms: Mapped[float] = mapped_column(Float)

    # Индекс по времени для аналитики
    timestamp_utc: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        default=lambda: datetime.now(timezone.utc), 
        index=True
    )