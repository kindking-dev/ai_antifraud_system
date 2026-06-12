"""
AI-POWERED BEHAVIORAL ANTI-FRAUD SERVICE: Main Scoring Controller.
Orchestrates Late Fusion (Matrix Veto) between Behavioral, Device SDK, and Transactional ML Engines.
Features High-Performance Real-Time Streaming Architecture via WebSockets + Redis Pub/Sub.
"""

import time
import json
import logging
import asyncio
from typing import Dict, Any, Optional, List

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Request, WebSocket, WebSocketDisconnect
from fastapi.security import APIKeyHeader

# --- Схемы (Contracts) ---
from app.schemas.transaction import FraudAnalysisRequest
from app.schemas.response import FraudAnalysisResponse, FraudAction, ReasonCode
from app.schemas.behavioral_model_service import BehavioralInferenceRequest, BehavioralInferenceResponse

# --- Инфраструктура ---
from app.core.config import settings
from app.repositories.pg_store import save_transaction_log
from app.repositories.redis_store import RedisStore

# --- Новые ML-движки (Production Grade) ---
from app.services.behavioral_engine import BehavioralEngine
from app.ml.inference.transaction_inference import predict_transaction_model

logger = logging.getLogger(__name__)
router = APIRouter()

API_KEY_HEADER = APIKeyHeader(name="X-API-KEY", auto_error=True)
SLA_LIMIT_MS = 50.0

# Легковесный движок, не требующий прогрева GPU
behavioral_engine = BehavioralEngine()

# =========================
# DEPENDENCIES
# =========================
async def verify_api_key(api_key: str = Depends(API_KEY_HEADER)) -> str:
    """Проверка API ключа (Zero Trust Network)."""
    if not api_key or api_key != settings.API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return api_key

def get_redis(request: Request) -> RedisStore:
    """Извлекает пул Redis из состояния FastAPI."""
    return request.app.state.redis_store


# =========================
# 1. BEHAVIORAL ENDPOINT (HTTP)
# =========================
@router.post(
    "/score-behavior",
    response_model=BehavioralInferenceResponse,
    dependencies=[Depends(verify_api_key)],
)
async def score_behavior(
    request_body: BehavioralInferenceRequest,
    request: Request
) -> BehavioralInferenceResponse:
    """Оценка 'на лету' биометрического профиля (Late Fusion State)."""
    start = time.perf_counter()
    uid = request_body.user_id
    redis_store = get_redis(request)

    try:
        events = getattr(request_body, 'events', [])
        if events:
            current_profile = behavioral_engine.extract_profile_from_events(events)
        elif getattr(request_body, 'features', None):
            current_profile = request_body.features.model_dump()
        else:
            current_profile = behavioral_engine.get_defaults_dict()

        baseline_raw = await redis_store.client.hget(f"user:{uid}:profile", "baseline")
        baseline_profile = json.loads(baseline_raw) if baseline_raw else {}

        # 🔥 AUTO-BASELINE: If no baseline exists, save the current profile as baseline.
        # This prevents 100% false positives on the first transaction.
        if not baseline_profile:
            baseline_profile = current_profile
            await redis_store.client.hset(f"user:{uid}:profile", "baseline", json.dumps(current_profile))

        risk_score = behavioral_engine.calculate_risk(baseline_profile, current_profile)

        # Запись состояния для Late Fusion (TTL 10 минут)
        state_key = f"user:{uid}:state"
        await redis_store.client.hset(state_key, "latest_behavior_score", float(risk_score))
        await redis_store.client.expire(state_key, 600)

        is_anomaly = risk_score > 0.75
        latency_ms = (time.perf_counter() - start) * 1000

        if latency_ms > SLA_LIMIT_MS:
            logger.warning(f"Behavioral SLA breach: {latency_ms:.2f} ms for user {uid}")

        return BehavioralInferenceResponse(
            user_id=uid,
            fraud_probability=round(risk_score, 4),
            is_anomaly=is_anomaly,
            processing_time_ms=round(latency_ms, 2),
            status="IMPOSTOR" if is_anomaly else "MATCH"
        )

    except Exception as e:
        logger.exception(f"CRITICAL BEHAVIORAL SCORING FAILURE for user {uid}: {e}")
        return BehavioralInferenceResponse(
            user_id=uid,
            fraud_probability=0.5,
            is_anomaly=False,
            processing_time_ms=0.0,
            status="ERROR"
        )


@router.post(
    "/set-baseline",
    dependencies=[Depends(verify_api_key)],
)
async def set_baseline(
    request_body: BehavioralInferenceRequest,
    request: Request
):
    """Сохраняет эталонный биометрический профиль для пользователя."""
    uid = request_body.user_id
    redis_store = get_redis(request)
    
    try:
        events = getattr(request_body, 'events', [])
        if events:
            current_profile = behavioral_engine.extract_profile_from_events(events)
        elif getattr(request_body, 'features', None):
            current_profile = request_body.features.model_dump()
        else:
            current_profile = behavioral_engine.get_defaults_dict()
        
        # Сохраняем в Redis
        await redis_store.client.hset(f"user:{uid}:profile", "baseline", json.dumps(current_profile))
        return {"status": "ok", "user_id": uid, "message": "Baseline saved successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



# =========================
# 2. TRANSACTION ENDPOINT (Late Fusion Matrix)
# =========================
@router.post(
    "/score-transaction",
    response_model=FraudAnalysisResponse,
    dependencies=[Depends(verify_api_key)],
)
async def score_transaction(
    request_body: FraudAnalysisRequest,
    request: Request,
    background_tasks: BackgroundTasks
) -> FraudAnalysisResponse:
    """CORE ENGINE: Мультисенсорный анализ (Late Fusion) + Матрица Вето."""
    start = time.perf_counter()
    uid = request_body.user_id
    tx_id = request_body.transaction_id
    redis_store = get_redis(request)

    try:
        payload = request_body.model_dump(mode="json")

        if settings.ENABLE_TELEMETRY_CONSOLE_LOGS:
            # Demo-only console output. Keep disabled during load tests.
            print("\n" + "🔴"*20 + " ALERT: INCOMING TELEMETRY " + "🔴"*20)
            print(f"📱 User ID: {uid} | Сумма: {payload.get('amount_kzt')} KZT")
            print("🧬 БИОМЕТРИЯ С ТЕЛЕФОНА:")
            print(json.dumps(payload.get('biometrics'), indent=4))
            print("🌐 СЕТЬ И УСТРОЙСТВО:")
            print(json.dumps(payload.get('network'), indent=4))
            print("🔴"*55 + "\n")

        # =========================
        # 🧠 MULTI-SENSOR FUSION
        # =========================
        redis_behavior_score = 0.0
        try:
            state_key = f"user:{uid}:state"
            raw_score = await redis_store.client.hget(state_key, "latest_behavior_score")
            if raw_score is not None:
                redis_behavior_score = float(raw_score)
        except Exception as e:
            logger.warning(f"Redis fetch failed for user {uid}: {e}")

        device_risk_score = 1.0 - request_body.session_trust_score
        behavior_score = max(redis_behavior_score, device_risk_score)
        tx_prob = float(predict_transaction_model(payload))

        # =========================
        # 🛡️ DECISION ENGINE (Matrix Veto)
        # =========================
        action = FraudAction.ALLOW
        reasons = []

        if behavior_score >= 0.75 or tx_prob >= 0.85:
            action = FraudAction.BLOCK
            if behavior_score >= 0.75:
                reasons.append(ReasonCode.SUSPICIOUS_BEHAVIOR)
            if tx_prob >= 0.85:
                reasons.append(ReasonCode.HIGH_ML_RISK)

        if action != FraudAction.BLOCK:
            if tx_prob >= 0.60 and behavior_score >= 0.60:
                action = FraudAction.BLOCK
                reasons.extend([ReasonCode.ELEVATED_RISK, ReasonCode.SUSPICIOUS_BEHAVIOR])
            elif tx_prob >= 0.50:
                action = FraudAction.CHALLENGE
                reasons.append(ReasonCode.ELEVATED_RISK)
            elif behavior_score >= 0.65:
                action = FraudAction.CHALLENGE
                reasons.append(ReasonCode.SUSPICIOUS_BEHAVIOR)

        reasons = list(set(reasons))
        final_fused_prob = float(max(tx_prob, behavior_score))
        latency_ms = (time.perf_counter() - start) * 1000

        background_tasks.add_task(
            save_transaction_log,
            {
                "transaction_id": tx_id,
                "user_id": uid,
                "amount_kzt": request_body.amount_kzt,
                "fraud_probability": final_fused_prob,
                "action": action.value,
                "processing_time_ms": latency_ms,
                "timestamp_utc": request_body.timestamp_utc
            }
        )

        return FraudAnalysisResponse(
            transaction_id=tx_id,
            action=action,
            fraud_probability=round(final_fused_prob, 4),
            reason_codes=reasons,
            feature_impacts={
                "behavior_score_impact": round(behavior_score, 4), 
                "tx_model_impact": round(tx_prob, 4)
            },
            processing_time_ms=round(latency_ms, 2)
        )

    except Exception as e:
        logger.exception(f"CRITICAL TRANSACTION SCORING FAILURE | Tx: {tx_id}: {e}")
        return FraudAnalysisResponse(
            transaction_id=tx_id,
            action=FraudAction.ALLOW,
            fraud_probability=0.0,
            reason_codes=[],
            feature_impacts={},
            processing_time_ms=0.0
        )


# =========================
# 3. TRANSACTION HISTORY (PostgreSQL Read)
# =========================
from app.repositories.pg_store import AsyncSessionLocal
from app.models.db_entities import TransactionLog
from sqlalchemy import select, delete

@router.delete(
    "/system/reset",
    dependencies=[Depends(verify_api_key)],
)
async def reset_system(request: Request):
    """Очищает базу данных PostgreSQL и кэш Redis для чистого демо."""
    redis_store = get_redis(request)
    try:
        async with AsyncSessionLocal() as session:
            await session.execute(delete(TransactionLog))
            await session.commit()
            
        await redis_store.client.flushall()
        
        return {"status": "ok", "message": "System reset successfully"}
    except Exception as e:
        logger.exception(f"Failed to reset system: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/transactions",
    dependencies=[Depends(verify_api_key)],
)
async def get_transactions(limit: int = 100) -> List[Dict[str, Any]]:
    """Fetch transaction audit trail from PostgreSQL for the Streamlit dashboard."""
    try:
        async with AsyncSessionLocal() as session:
            stmt = select(TransactionLog).order_by(TransactionLog.timestamp_utc.desc()).limit(limit)
            result = await session.execute(stmt)
            rows = result.scalars().all()
            return [
                {
                    "transaction_id": r.transaction_id,
                    "user_id": r.user_id,
                    "amount_kzt": r.amount_kzt,
                    "fraud_probability": r.fraud_probability,
                    "action": r.action,
                    "reason_codes": r.reason_codes or [],
                    "feature_impacts": r.feature_impacts or {},
                    "processing_time_ms": r.processing_time_ms,
                    "timestamp_utc": r.timestamp_utc.isoformat() if r.timestamp_utc else None,
                }
                for r in rows
            ]
    except Exception as e:
        logger.exception(f"Failed to fetch transactions: {e}")
        return []


# =========================
# 4. LIVE TELEMETRY STREAM (WebSockets & Redis Pub/Sub)
# =========================

@router.websocket("/ws/telemetry/client")
async def websocket_client_endpoint(websocket: WebSocket):
    """
    HOT PATH: Mobile client streams live telemetry here. Broadcasts to Redis Pub/Sub.
    Features Rate Limiting for DDoS protection.
    """
    await websocket.accept()
    redis_store = websocket.app.state.redis_store
    
    # Anti-Flood Configuration: Max ~30 frames per second
    RATE_LIMIT_SEC = 0.033 
    last_publish_time = 0.0

    try:
        while True:
            payload_text = await websocket.receive_text()
            now = time.perf_counter()
            
            # Drop frames if they arrive too fast to prevent Redis OOM
            if now - last_publish_time < RATE_LIMIT_SEC:
                continue
                
            last_publish_time = now
            
            # Zero-latency push to all inspectors
            await redis_store.client.publish("channel:telemetry:live", payload_text)
            # Cache the latest state for new connections
            await redis_store.client.set("live_telemetry_state", payload_text, ex=10)
            
    except WebSocketDisconnect:
        logger.info("Mobile client disconnected from telemetry stream.")
    except Exception as e:
        logger.error(f"WebSocket client error: {e}")


@router.websocket("/ws/telemetry/inspector")
async def websocket_inspector_endpoint(websocket: WebSocket):
    """
    HOT PATH: Deep Inspector subscribes to Redis Pub/Sub for zero-latency telemetry.
    Safely cancels async tasks on disconnect to prevent memory leaks.
    """
    await websocket.accept()
    redis_store = websocket.app.state.redis_store
    pubsub = redis_store.client.pubsub()
    
    try:
        await pubsub.subscribe("channel:telemetry:live")
        
        # Send initial state immediately
        last_state = await redis_store.client.get("live_telemetry_state")
        if last_state:
            text_data = last_state.decode('utf-8') if isinstance(last_state, bytes) else last_state
            await websocket.send_text(text_data)

        # Concurrency: Listen to Redis and WebSocket disconnects simultaneously
        async def read_from_redis():
            async for message in pubsub.listen():
                if message["type"] == "message":
                    data = message["data"]
                    text_data = data.decode('utf-8') if isinstance(data, bytes) else data
                    await websocket.send_text(text_data)

        async def read_from_ws():
            while True:
                await websocket.receive_text()

        redis_task = asyncio.create_task(read_from_redis())
        ws_task = asyncio.create_task(read_from_ws())

        done, pending = await asyncio.wait(
            [redis_task, ws_task], 
            return_when=asyncio.FIRST_COMPLETED
        )
        
        # Memory Leak Protection: Cancel hanging tasks
        for task in pending:
            task.cancel()

    except WebSocketDisconnect:
        logger.info("Inspector dashboard disconnected.")
    except Exception as e:
        logger.error(f"WebSocket inspector error: {e}")
    finally:
        await pubsub.unsubscribe("channel:telemetry:live")
        await pubsub.close()
