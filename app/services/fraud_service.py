import logging
import time
import uuid
from typing import Dict, Any

from app.ml.pipelines.behavioral_pipeline import behavioral_score
from app.ml.inference.transaction_inference import predict_transaction_model
from app.repositories.redis_store import update_behavioral_state

logger = logging.getLogger(__name__)


# =========================
# CONFIG
# =========================

TX_WEIGHT = 0.7
BEHAVIOR_WEIGHT = 0.3

DECLINE_THRESHOLD = 0.8
REVIEW_THRESHOLD = 0.5

MAX_ALLOWED_LATENCY_MS = 50


# =========================
# UTILS
# =========================

def _safe_float(v: Any) -> float:
    try:
        return float(v)
    except Exception:
        return 0.0


def _clamp_score(score: float) -> float:
    return max(0.0, min(1.0, score))


def _now_ms() -> int:
    return int(time.time() * 1000)


# =========================
# DECISION
# =========================

def _make_decision(score: float) -> str:
    if score >= DECLINE_THRESHOLD:
        return "decline"
    elif score >= REVIEW_THRESHOLD:
        return "review"
    return "approve"


def _risk_level(score: float) -> str:
    if score >= DECLINE_THRESHOLD:
        return "high"
    elif score >= REVIEW_THRESHOLD:
        return "medium"
    return "low"


# =========================
# VALIDATION
# =========================

def _validate_input(user_id: str, amount: float):
    if not user_id:
        raise ValueError("user_id is required")

    if amount is None:
        raise ValueError("amount is required")

    if amount < 0:
        raise ValueError("amount cannot be negative")


# =========================
# CORE SERVICE
# =========================

def fraud_score(
    user_id: str,
    amount: float,
    tx_features: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Production-grade fraud scoring service.

    Flow:
        1. Validate input
        2. Transaction model
        3. Behavioral model
        4. Fusion
        5. Decision
        6. Update behavioral state (async-safe)
        7. Return structured response

    SLA target: <50ms
    """

    trace_id = str(uuid.uuid4())
    start_time = time.perf_counter()

    try:
        _validate_input(user_id, amount)

        # =========================
        # 1. TRANSACTION MODEL
        # =========================
        tx_score = 0.0
        try:
            tx_score = _safe_float(
                predict_transaction_model(tx_features)
            )
        except Exception as e:
            logger.exception(f"[{trace_id}] TX model failed: {e}")

        # =========================
        # 2. BEHAVIORAL MODEL
        # =========================
        beh_score = 0.0
        try:
            beh_score = _safe_float(
                behavioral_score(user_id, amount)
            )
        except Exception as e:
            logger.exception(f"[{trace_id}] Behavioral failed: {e}")

        # =========================
        # 3. FUSION
        # =========================
        final_score = _clamp_score(
            TX_WEIGHT * tx_score +
            BEHAVIOR_WEIGHT * beh_score
        )

        # =========================
        # 4. DECISION
        # =========================
        decision = _make_decision(final_score)

        # =========================
        # 5. UPDATE STATE (НЕ БЛОКИРУЕТ)
        # =========================
        try:
            update_behavioral_state(
                user_id=user_id,
                amount=amount,
                timestamp=_now_ms()
            )
        except Exception as e:
            logger.warning(f"[{trace_id}] Redis update failed: {e}")

        # =========================
        # 6. LATENCY
        # =========================
        latency_ms = int((time.perf_counter() - start_time) * 1000)

        if latency_ms > MAX_ALLOWED_LATENCY_MS:
            logger.warning(
                f"[{trace_id}] Slow scoring: {latency_ms} ms"
            )

        # =========================
        # 7. OUTPUT
        # =========================
        return {
            "trace_id": trace_id,

            "score": float(final_score),
            "decision": decision,
            "risk_level": _risk_level(final_score),

            "transaction_score": float(tx_score),
            "behavioral_score": float(beh_score),

            "weights": {
                "transaction": TX_WEIGHT,
                "behavioral": BEHAVIOR_WEIGHT,
            },

            "latency_ms": latency_ms,
        }

    except Exception as e:
        logger.exception(f"[{trace_id}] FATAL fraud scoring error: {e}")

        return {
            "trace_id": trace_id,

            "score": 0.0,
            "decision": "approve",
            "risk_level": "low",

            "transaction_score": 0.0,
            "behavioral_score": 0.0,

            "weights": {
                "transaction": TX_WEIGHT,
                "behavioral": BEHAVIOR_WEIGHT,
            },

            "latency_ms": int((time.perf_counter() - start_time) * 1000),
        }


# =========================
# EXTENDED VERSION
# =========================

def fraud_score_full(
    user_id: str,
    amount: float,
    tx_features: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Расширенная версия (для логов / аналитики / SHAP)
    """

    result = fraud_score(user_id, amount, tx_features)

    # сюда потом добавим explainability
    result["debug"] = {
        "timestamp": _now_ms(),
        "model_version": "v1",
    }

    return result


# =========================
# DEBUG LOCAL RUN
# =========================

if __name__ == "__main__":
    test_tx = {
        "TransactionAmt": 120.5,
        "card1": 1234,
        "card2": 111,
        "addr1": 100,
        "P_emaildomain": "gmail.com",
    }

    result = fraud_score(
        user_id="test_user",
        amount=120.5,
        tx_features=test_tx,
    )

    print(result)