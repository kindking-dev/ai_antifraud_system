"""
Database ORM Models using SQLAlchemy 2.0.
Defines the schema for the Audit Trail.
"""

from datetime import datetime, timezone
from typing import Optional, List
from sqlalchemy import String, Float, DateTime, JSON
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy ORM models."""

    pass


class TransactionLog(Base):
    """
    Аудит-лог транзакций. Обязателен для compliance в финтехе.
    Хранит результаты скоринга и SHAP-объяснения.
    """

    __tablename__ = "transaction_logs"

    # transaction_id приходит от клиента, поэтому используем String
    transaction_id: Mapped[str] = mapped_column(String(50), primary_key=True)

    # Индекс для быстрого поиска истории конкретного юзера (O(log N))
    user_id: Mapped[str] = mapped_column(String(50), index=True)

    amount_kzt: Mapped[float] = mapped_column(Float)
    fraud_probability: Mapped[float] = mapped_column(Float)
    action: Mapped[str] = mapped_column(String(20))

    # JSON-колонка для списка причин (SHAP reason codes)
    reason_codes: Mapped[Optional[List[str]]] = mapped_column(JSON, nullable=True)

    processing_time_ms: Mapped[float] = mapped_column(Float)

    # Индекс по времени для построения графиков и отчетов
    timestamp_utc: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True
    )
