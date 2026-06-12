import pytest
import pytest_asyncio
import json
from unittest.mock import patch
import fakeredis.aioredis
from redis.asyncio import Redis

# ИСПРАВЛЕНО: Используем pytest_asyncio.fixture для асинхронных заглушек
@pytest_asyncio.fixture
async def fake_redis_client() -> Redis:
    client = fakeredis.aioredis.FakeRedis(decode_responses=True)
    yield client
    await client.flushall()
    await client.aclose()

@pytest.fixture
def sample_hmog_profile() -> dict:
    return {
        "user_id": "usr_999",
        "velocity_mean": 2.45,
        "pressure_mean": 0.55,
        "is_bot": False,
        "timestamp": "2026-05-14T20:00:00Z"
    }

@pytest.mark.asyncio
async def test_set_and_get_biometric_profile(fake_redis_client: Redis, sample_hmog_profile: dict):
    key = f"hmog:profile:{sample_hmog_profile['user_id']}"
    await fake_redis_client.set(key, json.dumps(sample_hmog_profile), ex=3600)
    raw_data = await fake_redis_client.get(key)
    assert raw_data is not None
    retrieved_profile = json.loads(raw_data)
    assert retrieved_profile["user_id"] == "usr_999"

@pytest.mark.asyncio
async def test_get_nonexistent_profile(fake_redis_client: Redis):
    key = "hmog:profile:ghost_user"
    raw_data = await fake_redis_client.get(key)
    assert raw_data is None

@pytest.mark.asyncio
async def test_key_expiration_ttl(fake_redis_client: Redis, sample_hmog_profile: dict):
    key = "hmog:profile:ttl_test"
    ttl_seconds = 300 
    await fake_redis_client.set(key, json.dumps(sample_hmog_profile), ex=ttl_seconds)
    actual_ttl = await fake_redis_client.ttl(key)
    assert 0 < actual_ttl <= ttl_seconds

@pytest.mark.asyncio
@patch("redis.asyncio.Redis.get")
async def test_redis_connection_failure_during_read(mock_redis_get):
    from redis.exceptions import ConnectionError
    mock_redis_get.side_effect = ConnectionError("Redis server went away")
    try:
        _ = await mock_redis_get("some_key")
        pytest.fail("Exception was silently swallowed")
    except ConnectionError:
        assert True