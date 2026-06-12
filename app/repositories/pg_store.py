"""
PostgreSQL Asynchronous Repository.
Handles background persistence for the Audit Trail.
"""

import structlog
from typing import Dict, Any
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlalchemy.exc import SQLAlchemyError

from app.core.config import settings
from app.models.db_entities import TransactionLog

logger = structlog.get_logger(__name__)

# Создаем асинхронный движок с пулом соединений
engine = create_async_engine(
    settings.SQLALCHEMY_DATABASE_URI,
    pool_size=20,
    max_overflow=10,
    pool_pre_ping=True,
)

# Фабрика сессий
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)


async def save_transaction_log(log_data: Dict[str, Any]) -> None:
    """
    Saves the scoring result to PostgreSQL.
    Designed to be run as a FastAPI BackgroundTask.
    """
    try:
        async with AsyncSessionLocal() as session:
            async with session.begin():
                # Создаем ORM-объект из переданного словаря
                db_record = TransactionLog(**log_data)
                session.add(db_record)
            
            # Commit происходит автоматически при выходе из context manager 'begin'
            logger.info(
                "audit_trail_saved", 
                transaction_id=log_data.get("transaction_id")
            )

    except SQLAlchemyError as e:
        logger.error(
            "db_persistence_failed",
            error=str(e),
            transaction_id=log_data.get("transaction_id"),
        )
    except Exception as e:
        logger.exception(
            "unexpected_persistence_error",
            error=str(e),
            transaction_id=log_data.get("transaction_id")
        )