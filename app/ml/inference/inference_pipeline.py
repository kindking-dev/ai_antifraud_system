# app/ml/inference/inference_pipeline.py

import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional

from catboost import CatBoostClassifier

from app.ml.inference.transaction_inference import predict_transaction_model

logger = logging.getLogger(__name__)


class FraudInferencePipeline:
    """
    FULL HYBRID INFERENCE PIPELINE

    Flow:
        1. Transaction model
        2. Behavioral model
        3. Fusion
        4. Decision

    SLA target: <50ms
    """

    def __init__(
        self,
        model: Optional[CatBoostClassifier] = None,
        artifacts_dir: Optional[Path] = None,
    ):
        self.model = model

        self.tx_weight = 0.7
        self.beh_weight = 0.3

        self.block_threshold = 0.85
        self.challenge_threshold = 0.50

        logger.info("Hybrid inference pipeline initialized")

    # =========================
    # UTILS
    # =========================

    def _safe_float(self, v: Any) -> float:
        try:
            f = float(v)
            if f != f or f == float("inf") or f == float("-inf"):
                return 0.0
            return f
        except Exception:
            return 0.0

    def _clamp(self, v: float) -> float:
        return max(0.0, min(1.0, v))

    def _decision(self, score: float) -> str:
        if score >= self.block_threshold:
            return "BLOCK"
        elif score >= self.challenge_threshold:
            return "CHALLENGE"
        return "ALLOW"

    # =========================
    # MAIN SCORING
    # =========================

    async def score(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        start = time.perf_counter()

        try:
            user_id = payload.get("user_id", "unknown")
            amount = self._safe_float(payload.get("amount_kzt", 0.0))

            # =========================
            # 1. TRANSACTION MODEL
            # =========================
            tx_score = 0.0
            try:
                tx_score = self._safe_float(predict_transaction_model(payload))
            except Exception as e:
                logger.warning(f"TX model failed: {e}")

            # =========================
            # 2. BEHAVIORAL MODEL (LATE FUSION)
            # =========================
            beh_score = 0.0
            try:
                # Score was fetched from Redis and injected into payload by scoring.py
                beh_score = self._safe_float(payload.get("behavior_score", 0.5))
            except Exception as e:
                logger.warning(f"Behavioral failed: {e}")

            # =========================
            # 3. FUSION
            # =========================
            final_score = self._clamp(
                self.tx_weight * tx_score + self.beh_weight * beh_score
            )

            # =========================
            # 4. DECISION
            # =========================
            action = self._decision(final_score)

            latency = (time.perf_counter() - start) * 1000

            return {
                "fraud_probability": round(final_score, 4),
                "action": action,
                "transaction_score": round(tx_score, 4),
                "behavioral_score": round(beh_score, 4),
                "processing_time_ms": round(latency, 2),
                "status": "success",
            }

        except Exception as e:
            logger.exception("hybrid_inference_failed")

            return {
                "fraud_probability": 0.0,
                "action": "ALLOW",
                "transaction_score": 0.0,
                "behavioral_score": 0.0,
                "processing_time_ms": round((time.perf_counter() - start) * 1000, 2),
                "status": "error",
                "error": str(e),
            }

    # =========================
    # WARMUP
    # =========================

    async def warmup(self, payload: Dict[str, Any]):
        for _ in range(3):
            await self.score(payload)
