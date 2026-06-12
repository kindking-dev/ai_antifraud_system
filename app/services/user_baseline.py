# app/services/user_baseline.py

import logging
from typing import Dict, Any

from app.repositories.redis_store import RedisStore
from app.services.behavioral_engine import BehavioralEngine


logger = logging.getLogger(__name__)


# =========================
# CONFIG
# =========================

EMA_ALPHA = 0.1
EPS = 1e-6

BASELINE_TTL_SEC = 60 * 60 * 24 * 7  # 7 дней


# 🔥 строго совпадает с BehavioralEngine
BASELINE_FEATURES = BehavioralEngine.get_feature_names()


# =========================
# UTIL
# =========================

def _safe_float(v: Any) -> float:
    try:
        if v is None:
            return 0.0
        return float(v)
    except Exception:
        return 0.0


def _key(user_id: str) -> str:
    return f"user:{user_id}:baseline"


# =========================
# UPDATE BASELINE
# =========================

async def update_baseline(user_id: str, features: Dict[str, Any]) -> None:
    """
    EMA baseline (online learning)
    """

    redis_store = RedisStore()
    client = redis_store.client

    try:
        key = _key(user_id)

        pipe = client.pipeline(transaction=False)

        pipe.hincrby(key, "n", 1)

        for feat in BASELINE_FEATURES:
            value = _safe_float(features.get(feat, 0.0))

            mean_key = f"{feat}:mean"
            var_key = f"{feat}:var"

            prev_mean = await client.hget(key, mean_key)
            prev_var = await client.hget(key, var_key)

            if prev_mean is None:
                pipe.hset(key, mean_key, value)
                pipe.hset(key, var_key, 0.0)
                continue

            prev_mean = float(prev_mean)
            prev_var = float(prev_var) if prev_var else 0.0

            # EMA mean
            new_mean = (1 - EMA_ALPHA) * prev_mean + EMA_ALPHA * value

            # EMA variance
            diff = value - prev_mean
            new_var = (1 - EMA_ALPHA) * prev_var + EMA_ALPHA * (diff ** 2)

            pipe.hset(key, mean_key, new_mean)
            pipe.hset(key, var_key, new_var)

        # TTL (очень важно)
        pipe.expire(key, BASELINE_TTL_SEC)

        await pipe.execute()

    except Exception as e:
        logger.error(f"baseline_update_error: {e}")


# =========================
# Z-SCORE
# =========================

async def compute_zscore(user_id: str, features: Dict[str, Any]) -> Dict[str, float]:

    redis_store = RedisStore()
    client = redis_store.client

    zscores: Dict[str, float] = {}

    try:
        key = _key(user_id)

        for feat in BASELINE_FEATURES:
            value = _safe_float(features.get(feat, 0.0))

            mean_raw = await client.hget(key, f"{feat}:mean")
            var_raw = await client.hget(key, f"{feat}:var")

            if mean_raw is None or var_raw is None:
                zscores[feat] = 0.0
                continue

            mean = float(mean_raw)
            var = float(var_raw)

            std = (var ** 0.5) if var > 0 else EPS

            z = (value - mean) / (std + EPS)

            # clamp
            z = max(min(z, 10.0), -10.0)

            zscores[feat] = float(z)

        return zscores

    except Exception as e:
        logger.error(f"zscore_error: {e}")
        return {feat: 0.0 for feat in BASELINE_FEATURES}


# =========================
# AGGREGATION
# =========================

def aggregate_zscore(zscores: Dict[str, float]) -> float:
    if not zscores:
        return 0.0

    try:
        values = [abs(v) for v in zscores.values()]

        if not values:
            return 0.0

        mean_z = sum(values) / len(values)

        # нормализация
        score = min(mean_z / 5.0, 1.0)

        return float(score)

    except Exception:
        return 0.0


# =========================
# FULL PIPELINE
# =========================

async def baseline_anomaly_score(user_id: str, features: Dict[str, Any]) -> float:
    """
    1. Z-score
    2. Aggregate
    3. Update baseline (post-score)
    """

    try:
        zscores = await compute_zscore(user_id, features)

        score = aggregate_zscore(zscores)

        # 🔥 ВАЖНО: update ПОСЛЕ
        await update_baseline(user_id, features)

        return float(score)

    except Exception as e:
        logger.error(f"baseline_pipeline_error: {e}")
        return 0.0


# =========================
# DEBUG
# =========================

if __name__ == "__main__":
    import asyncio

    async def test():
        test_features = {
            "tx_count": 5,
            "tx_last_1min": 2,
            "tx_last_5min": 4,
            "avg_amount": 120.5,
            "last_amount": 150.0,
            "velocity_sec": 0.1,
        }

        score = await baseline_anomaly_score("test_user", test_features)
        print("Score:", score)

    asyncio.run(test())