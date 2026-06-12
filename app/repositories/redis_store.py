import logging
import time
import uuid
from typing import Dict, Any, Optional

import redis.asyncio as redis
from redis.asyncio import ConnectionPool

from app.core.config import settings

logger = logging.getLogger(__name__)


# =========================
# CONSTANTS 
# =========================

WINDOW_1M = 60.0
WINDOW_5M = 300.0

TTL_SECONDS = 86400  # 24h


class RedisStore:
    """
    Production-grade Redis behavioral store.

    Guarantees:
    - strict train/inference parity
    - NO velocity inversion (store raw time_delta only)
    - O(log N) ZSET ops
    - TTL-safe memory
    - SLA-safe timeouts
    - fail-safe fallback
    """

    def __init__(self, host=None, port=None, socket_timeout=None, **kwargs):
        self.pool = ConnectionPool(
            host=host or settings.REDIS_HOST,
            port=port or settings.REDIS_PORT,
            db=settings.REDIS_DB,
            decode_responses=True,
            socket_timeout=socket_timeout or settings.REDIS_SOCKET_TIMEOUT,
            socket_connect_timeout=socket_timeout or settings.REDIS_SOCKET_TIMEOUT,
            max_connections=getattr(settings, "REDIS_MAX_CONNECTIONS", 50),
            retry_on_timeout=True,
        )

        self.client = redis.Redis(connection_pool=self.pool)

    # =========================
    # UPDATE (POST-SCORING ONLY)
    # =========================

    async def update_behavioral(
        self,
        user_id: str,
        amount: float,
        timestamp: Optional[float] = None,
    ) -> None:
        """
        STRICT: call AFTER scoring

        Stores:
        - tx_times (ZSET)
        - tx_count
        - total_amount
        - last_amount
        - last_time
        - time_delta (CRITICAL)
        """

        now = float(timestamp or time.time())
        key = f"user:{user_id}"

        try:
            # --- 1. get last_time BEFORE pipeline ---
            last_time_raw = await self.client.get(f"{key}:last_time")

            if last_time_raw:
                last_time = float(last_time_raw)
                time_delta = max(now - last_time, 1.0)
            else:
                time_delta = 9999.0  # first event → low risk

            # --- 2. pipeline ---
            pipe = self.client.pipeline(transaction=False)

            # unique event
            event_id = f"{now}:{uuid.uuid4().hex[:6]}"
            pipe.zadd(f"{key}:tx_times", {event_id: now})

            # cleanup old events
            pipe.zremrangebyscore(f"{key}:tx_times", 0, now - WINDOW_5M)

            # counters
            pipe.incr(f"{key}:tx_count")
            pipe.set(f"{key}:last_amount", float(amount))
            pipe.incrbyfloat(f"{key}:total_amount", float(amount))

            # time tracking
            pipe.set(f"{key}:last_time", now)
            pipe.set(f"{key}:time_delta", time_delta)  # ✅ CRITICAL

            # TTL
            pipe.expire(f"{key}:tx_times", TTL_SECONDS)
            pipe.expire(f"{key}:tx_count", TTL_SECONDS)
            pipe.expire(f"{key}:last_amount", TTL_SECONDS)
            pipe.expire(f"{key}:total_amount", TTL_SECONDS)
            pipe.expire(f"{key}:last_time", TTL_SECONDS)
            pipe.expire(f"{key}:time_delta", TTL_SECONDS)

            await pipe.execute()

        except Exception as e:
            logger.warning(f"redis_update_failed user={user_id} err={e}")

    # =========================
    # READ (FOR INFERENCE)
    # =========================

    async def get_behavioral(self, user_id: str) -> Dict[str, Any]:
        """
        Returns RAW features (NO amplification, NO velocity inversion)
        """

        now = float(time.time())
        key = f"user:{user_id}"

        try:
            pipe = self.client.pipeline(transaction=False)

            pipe.get(f"{key}:tx_count")
            pipe.zcount(f"{key}:tx_times", now - WINDOW_1M, now)
            pipe.zcount(f"{key}:tx_times", now - WINDOW_5M, now)
            pipe.get(f"{key}:total_amount")
            pipe.get(f"{key}:last_amount")
            pipe.get(f"{key}:time_delta")  # ✅ CRITICAL

            res = await pipe.execute()

            tx_count = float(res[0]) if res[0] else 0.0
            tx_1m = float(res[1]) if res[1] else 0.0
            tx_5m = float(res[2]) if res[2] else 0.0
            total_amt = float(res[3]) if res[3] else 0.0
            last_amt = float(res[4]) if res[4] else 0.0
            time_delta = float(res[5]) if res[5] else 9999.0

            avg_amount = total_amt / max(tx_count, 1.0)

            return {
                "tx_count": tx_count,
                "tx_last_1min": tx_1m,
                "tx_last_5min": tx_5m,
                "avg_amount": avg_amount,
                "last_amount": last_amt,
                "time_delta": time_delta,  # ✅ NOT velocity
            }

        except Exception as e:
            logger.warning(f"redis_read_failed user={user_id} err={e}")

            return {
                "tx_count": 0.0,
                "tx_last_1min": 0.0,
                "tx_last_5min": 0.0,
                "avg_amount": 0.0,
                "last_amount": 0.0,
                "time_delta": 9999.0,
            }

    # =========================
    # CLOSE
    # =========================

    async def close(self):
        if self.pool:
            await self.pool.disconnect()
            logger.info("redis_closed")