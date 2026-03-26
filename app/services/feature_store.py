"""
Real-time Feature Store Module using Redis.
Calculates behavioral aggregates (Velocity, Spikes) with O(1) complexity.
"""

import structlog
import redis.asyncio as redis
from typing import Dict, Any

logger = structlog.get_logger(__name__)

# Асинхронный пул соединений (High-throughput)
redis_client = redis.Redis(
    host="localhost",
    port=6379,
    db=0,
    decode_responses=True,
    socket_timeout=0.1,  # Строгий таймаут 100мс для защиты SLA
)


async def get_and_update_user_profile(
    user_id: str, current_amount: float
) -> Dict[str, Any]:
    """
    Извлекает историю пользователя и обновляет ее атомарно через Redis Pipeline.
    """
    velocity_key = f"user:{user_id}:velocity_24h"

    try:
        # Pipeline позволяет отправить все команды за 1 сетевой запрос
        pipe = redis_client.pipeline()

        # 1. Читаем текущее значение
        pipe.get(velocity_key)
        # 2. Увеличиваем счетчик транзакций на +1
        pipe.incr(velocity_key)
        # 3. Устанавливаем время жизни ключа (24 часа = 86400 сек)
        pipe.expire(velocity_key, 86400)

        # Выполняем транзакцию
        results = await pipe.execute()

        # Результат GET (до инкремента) находится в results[0]
        current_count = float(results[0]) if results[0] else 0.0

        # Логика: если это уже 4-я транзакция за день, помечаем как аномалию (Spike)
        is_spike = 1 if current_count >= 4 else 0

        logger.info(
            "feature_store_hit", user_id=user_id, previous_tx_count=int(current_count)
        )

        return {
            "Velocity_24h_Count": current_count,
            "Is_Velocity_Spike": is_spike,
            # Статические параметры карты (в реальном проекте берутся из БД)
            "card1": 10000,
            "card2": 500,
            "C1": 1.0,
            "C2": 1.0,
            "V1": 1.0,
            "V2": 1.0,
        }

    except Exception as e:
        logger.error("redis_connection_failed", error=str(e))
        # Fallback (Circuit Breaker): если Redis упал, пропускаем транзакцию по нулям, чтобы не блочить весь банк
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
