"""
Scoring API Router with Explainable AI (SHAP) integration.
"""

import time
import structlog
from catboost import Pool
from fastapi import APIRouter, HTTPException

from app.schemas.transaction import FraudAnalysisRequest
from app.schemas.response import FraudAnalysisResponse, FraudAction, ReasonCode
from app.core.state import ml_models

logger = structlog.get_logger(__name__)
router = APIRouter()

FEATURE_NAMES = [
    "amount_kzt",
    "Velocity_24h_Count",
    "Is_Velocity_Spike",
    "Sensor_Keystroke_Variance",
    "Device_Trust_Score",
    "card1",
    "card2",
    "C1",
    "C2",
    "V1",
    "V2",
]


async def fetch_redis_features(user_id: str) -> dict:
    """Mock for O(1) Redis Feature Store lookup."""
    return {
        "Velocity_24h_Count": 15.0,
        "Is_Velocity_Spike": 0,
        "Sensor_Keystroke_Variance": 0.02,
        "Device_Trust_Score": 0.35,
        "card1": 10000,
        "card2": 500,
        "C1": 1.0,
        "C2": 1.0,
        "V1": 1.0,
        "V2": 1.0,
    }


@router.post("/score-transaction", response_model=FraudAnalysisResponse)
async def score_transaction(request: FraudAnalysisRequest) -> FraudAnalysisResponse:
    start_time = time.perf_counter()
    logger.info("processing_transaction", transaction_id=request.transaction_id)

    try:
        model = ml_models.get("core_scorer")
        if not model:
            raise HTTPException(status_code=500, detail="ML Model not initialized")

        hist_features = await fetch_redis_features(request.user_id)

        feature_vector = [
            request.amount_kzt,
            hist_features["Velocity_24h_Count"],
            hist_features["Is_Velocity_Spike"],
            hist_features["Sensor_Keystroke_Variance"],
            hist_features["Device_Trust_Score"],
            hist_features["card1"],
            hist_features["card2"],
            hist_features["C1"],
            hist_features["C2"],
            hist_features["V1"],
            hist_features["V2"],
        ]

        # Inference
        fraud_prob = float(model.predict_proba([feature_vector])[0][1])

        # Explainable AI (XAI) logic
        inference_pool = Pool(data=[feature_vector])
        shap_values = model.get_feature_importance(
            data=inference_pool, type="ShapValues"
        )[0]

        reasons = []
        action = FraudAction.ALLOW

        if fraud_prob > 0.50:
            action = FraudAction.BLOCK if fraud_prob > 0.85 else FraudAction.CHALLENGE
            contributions = dict(zip(FEATURE_NAMES, shap_values[:-1]))
            top_drivers = sorted(
                contributions.items(), key=lambda x: x[1], reverse=True
            )

            for feature, impact in top_drivers:
                if impact <= 0.05:
                    break

                if feature in ["Sensor_Keystroke_Variance", "Device_Trust_Score"]:
                    reasons.append(ReasonCode.SUSPICIOUS_BEHAVIOR)
                elif feature in ["Velocity_24h_Count", "Is_Velocity_Spike"]:
                    reasons.append(ReasonCode.AML_VELOCITY_SPIKE)

                if len(reasons) >= 2:
                    break

            if not reasons:
                reasons.append(ReasonCode.HIGH_ML_RISK)

        proc_time = (time.perf_counter() - start_time) * 1000

        return FraudAnalysisResponse(
            transaction_id=request.transaction_id,
            action=action,
            fraud_probability=round(fraud_prob, 4),
            reason_codes=list(set(reasons)),
            processing_time_ms=round(proc_time, 2),
        )

    except Exception as e:
        logger.exception("scoring_failed", error=str(e))
        raise HTTPException(status_code=500, detail="Internal ML engine error")
