"""
PostgreSQL Asynchronous Repository.
Handles background persistence for the Audit Trail.
"""

import structlog
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlalchemy.exc import SQLAlchemyError

from app.core.config import settings
from app.models.db_entities import TransactionLog

logger = structlog.get_logger(__name__)

# Создаем асинхронный движок с пулом соединений для высоких нагрузок
engine = create_async_engine(
    settings.SQLALCHEMY_DATABASE_URI,
    pool_size=20,
    max_overflow=10,
    # echo=False # Поставь True, если захочешь видеть SQL-запросы в консоли
)

# Фабрика сессий
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)


async def save_transaction_log(log_data: dict) -> None:
    """
    Saves the scoring result to PostgreSQL.
    Designed to be run as a FastAPI BackgroundTask.
    """
    try:
        async with AsyncSessionLocal() as session:
            # Создаем ORM-объект из словаря
            db_record = TransactionLog(**log_data)
            session.add(db_record)
            await session.commit()

            logger.info("audit_trail_saved", transaction_id=db_record.transaction_id)

    except SQLAlchemyError as e:
        # Если БД упала, мы просто логируем ошибку, но не роняем приложение
        logger.error(
            "db_persistence_failed",
            error=str(e),
            transaction_id=log_data.get("transaction_id"),
        )
