import asyncio
import pandas as pd
from pathlib import Path
import structlog
import redis.asyncio as redis

logger = structlog.get_logger(__name__)

BASE_DIR = Path(__file__).resolve().parent
PROFILES_PATH = BASE_DIR / "data" / "processed" / "user_profiles.parquet"

# Настройки Redis (стандартные для твоего docker-compose)
REDIS_HOST = "localhost"
REDIS_PORT = 6379

async def load_profiles():
    logger.info("🚀 СТАРТ ЗАГРУЗКИ ПРОФИЛЕЙ В REDIS")
    
    if not PROFILES_PATH.exists():
        logger.error(f"❌ Файл {PROFILES_PATH} не найден!")
        return

    df = pd.read_parquet(PROFILES_PATH)
    logger.info(f"Загружено {len(df)} профилей из Parquet.")

    # Подключаемся к Redis
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    
    loaded_count = 0
    for _, row in df.iterrows():
        user_id = str(int(row['user_id']))
        profile_key = f"user:{user_id}:profile"
        
        # Превращаем строку pandas в словарь (только фичи, без user_id)
        profile_data = row.drop('user_id').to_dict()
        
        # Сохраняем в Redis Hash (HSET)
        await r.hset(profile_key, mapping=profile_data)
        loaded_count += 1

    await r.close()
    logger.info(f"✅ В Redis успешно загружено {loaded_count} эталонных профилей.")

if __name__ == "__main__":
    asyncio.run(load_profiles())