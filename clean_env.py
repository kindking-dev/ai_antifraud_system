"""
Infrastructure Cleanup Utility.
Resets Redis Feature Store and PostgreSQL Audit Trail.
"""

import asyncio
import structlog
import redis.asyncio as redis
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text
from app.core.config import settings

logger = structlog.get_logger(__name__)


async def reset_infrastructure():
    print("🧹 Starting Global System Reset...")

    try:
        # 1. Сброс Redis (Feature Store)
        r = redis.Redis(host=settings.REDIS_HOST, port=settings.REDIS_PORT)
        await r.flushall()
        print("✅ Redis (redis-antifraud) has been cleared.")

        # 2. Очистка PostgreSQL (Audit Trail)
        engine = create_async_engine(settings.SQLALCHEMY_DATABASE_URI)
        async with engine.begin() as conn:
            # Очищаем таблицу и сбрасываем счетчик ID
            await conn.execute(
                text("TRUNCATE TABLE transaction_logs RESTART IDENTITY CASCADE;")
            )

        await engine.dispose()
        print("✅ PostgreSQL (transaction_logs) has been truncated.")
        print("\n🚀 SYSTEM READY FOR CLEAN DEMO")

    except Exception as e:
        print(f"❌ Error during reset: {e}")


if __name__ == "__main__":
    asyncio.run(reset_infrastructure())
