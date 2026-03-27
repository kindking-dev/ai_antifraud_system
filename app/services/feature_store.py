"""
Real-time Feature Store Module using Redis.
Calculates behavioral aggregates (Velocity, Spikes) with O(1) complexity.
"""

import structlog
import redis.asyncio as redis
from typing import Dict, Any
from app.core.config import settings

logger = structlog.get_logger(__name__)

# Асинхронный пул соединений (High-throughput)
# Параметры хоста и порта берутся из централизованных настроек[cite: 62, 63].
redis_client = redis.Redis(
    host=settings.REDIS_HOST,
    port=settings.REDIS_PORT,
    db=0,
    decode_responses=True,
    socket_timeout=0.1,  # Строгий таймаут 100мс для защиты SLA < 50ms.
)


async def get_and_update_user_profile(
    user_id: str, current_amount: float
) -> Dict[str, Any]:
    """
    Извлекает историю транзакционной активности пользователя и обновляет её.
    Использует Redis Pipeline для обеспечения атомарности и минимальной задержки[cite: 117].
    """
    velocity_key = f"user:{user_id}:velocity_24h"

    try:
        # Pipeline минимизирует количество сетевых задержек (RTT) [cite: 117]
        pipe = redis_client.pipeline()

        # 1. Чтение текущего значения счетчика
        pipe.get(velocity_key)
        # 2. Инкремент счетчика транзакций
        pipe.incr(velocity_key)
        # 3. Обновление времени жизни ключа (24 часа) [cite: 118]
        pipe.expire(velocity_key, 86400)

        # Выполнение всех команд за один асинхронный вызов
        results = await pipe.execute()

        # Значение счетчика до текущего инкремента
        current_count = float(results[0]) if results[0] else 0.0

        # Определение аномального всплеска активности (Velocity Spike)
        # В дипломном проекте порог установлен на уровне >= 5 транзакций[cite: 119].
        is_spike = 1 if current_count >= 5 else 0

        logger.info(
            "feature_store_hit", user_id=user_id, previous_tx_count=int(current_count)
        )

        return {
            "Velocity_24h_Count": current_count,
            "Is_Velocity_Spike": is_spike,
            # Статические параметры карты (в промышленной системе запрашиваются из БД)
            "card1": 10000,
            "card2": 500,
            "C1": 1.0,
            "C2": 1.0,
            "V1": 1.0,
            "V2": 1.0,
        }

    except Exception as e:
        # Реализация паттерна Circuit Breaker: явное логирование активации Fallback-режима
        logger.critical(
            "REDIS_UNAVAILABLE_FALLBACK_ACTIVE",
            error=str(e),
            user_id=user_id,
            strategy="PASS_WITH_ZERO_FEATURES",
        )

        # Безопасный возврат дефолтных значений: банк продолжает работу, не блокируя клиента[cite: 121, 122].
        return {
            "Velocity_24h_Count": 0.0,
            "Is_Velocity_Spike": 0,
            "card1": 10000,
            "card2": 500,
            "C1": 1.0,
            "C2": 1.0,
            "V1": 1.0,
            "V2": 1.0,
        }
