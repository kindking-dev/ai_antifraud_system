import asyncio
import redis.asyncio as aioredis
import json

async def fix_redis():
    r = aioredis.Redis(host='localhost', port=6379, decode_responses=True)
    
    print("🔍 Checking Redis keys...")
    keys = await r.keys("user:*:profile")
    print(f"Found {len(keys)} profiles: {keys[:5]}")

    # Создаем "Золотой профиль" для юзера 7 (эталонные значения)
    # Эти значения взяты как средние по датасету Touchalytics
    test_profile = {
        "duration_ms_mean": "250.0",
        "length_px_mean": "450.0",
        "velocity_mean": "1.2", # ОЧЕНЬ ВАЖНО: эталонная скорость
        "median_pressure_mean": "0.15",
        "median_area_mean": "0.3"
    }
    
    print("🛠️ Injecting reference profile for user 7...")
    await r.hset("user:7:profile", mapping=test_profile)
    
    # Проверка
    saved = await r.hgetall("user:7:profile")
    print(f"✅ Successfully verified profile 7: {saved}")
    await r.close()

if __name__ == "__main__":
    asyncio.run(fix_redis())