"""
Scoring API Router with Explainable AI (SHAP) and Stateful Redis integration.
"""

import time
import structlog
from catboost import Pool
from fastapi import APIRouter, HTTPException

from app.schemas.transaction import FraudAnalysisRequest
from app.schemas.response import FraudAnalysisResponse, FraudAction, ReasonCode
from app.core.state import ml_models
from app.services.feature_store import get_and_update_user_profile

logger = structlog.get_logger(__name__)
router = APIRouter()

# Strict feature order matching CatBoost training pool
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


@router.post("/score-transaction", response_model=FraudAnalysisResponse)
async def score_transaction(request: FraudAnalysisRequest) -> FraudAnalysisResponse:
    start_time = time.perf_counter()
    logger.info("processing_transaction", transaction_id=request.transaction_id)

    try:
        # 1. State Access
        model = ml_models.get("core_scorer")
        if not model:
            logger.error("ml_model_not_found_in_state")
            raise HTTPException(status_code=500, detail="ML Model not initialized")

        # 2. Stateful Feature Store Integration (Real Redis)
        # Атомарное чтение и обновление счетчиков пользователя
        hist_features = await get_and_update_user_profile(
            user_id=request.user_id, current_amount=request.amount_kzt
        )

        # 3. Dynamic payload extraction
        trust_score = (
            request.session_trust_score
            if request.session_trust_score is not None
            else 0.95
        )

        sensor_variance = 0.5
        if request.biometrics:
            if hasattr(request.biometrics, "touch_pressure_variance"):
                sensor_variance = request.biometrics.touch_pressure_variance
            elif isinstance(request.biometrics, dict):
                sensor_variance = request.biometrics.get("touch_pressure_variance", 0.5)

        # 4. Feature Vector Assembly
        feature_vector = [
            request.amount_kzt,
            hist_features["Velocity_24h_Count"],
            hist_features["Is_Velocity_Spike"],
            sensor_variance,
            trust_score,
            hist_features["card1"],
            hist_features["card2"],
            hist_features["C1"],
            hist_features["C2"],
            hist_features["V1"],
            hist_features["V2"],
        ]

        # 5. ML Inference & XAI
        fraud_prob = float(model.predict_proba([feature_vector])[0][1])

        # --- HYBRID RULES ENGINE (Бизнес-логика AML) ---
        # Перекрываем решение ML-модели, если нарушены жесткие правила безопасности
        is_hard_rule_triggered = False

        if hist_features["Is_Velocity_Spike"] == 1:
            logger.warning("aml_rule_triggered", rule="VELOCITY_SPIKE")
            fraud_prob = max(
                fraud_prob, 0.88
            )  # Искусственно завышаем риск до зоны BLOCK
            is_hard_rule_triggered = True

        if sensor_variance < 0.05:
            logger.warning("aml_rule_triggered", rule="BOT_BIOMETRICS")
            fraud_prob = max(
                fraud_prob, 0.75
            )  # Искусственно завышаем до зоны CHALLENGE
            is_hard_rule_triggered = True
        # -----------------------------------------------

        inference_pool = Pool(data=[feature_vector])
        shap_values = model.get_feature_importance(
            data=inference_pool, type="ShapValues"
        )[0]

        # 6. Business Logic & Thresholding
        reasons = []
        action = FraudAction.ALLOW

        if fraud_prob > 0.50:
            action = FraudAction.BLOCK if fraud_prob > 0.85 else FraudAction.CHALLENGE

            # Если сработало жесткое правило, SHAP нам не так важен, мы точно знаем причину
            if is_hard_rule_triggered:
                if hist_features["Is_Velocity_Spike"] == 1:
                    reasons.append(ReasonCode.AML_VELOCITY_SPIKE)
                if sensor_variance < 0.05:
                    reasons.append(ReasonCode.SUSPICIOUS_BEHAVIOR)
            else:
                # Обычный SHAP анализ
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

        # 6. Business Logic & Thresholding
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

        # 7. SLA Observability
        proc_time = (time.perf_counter() - start_time) * 1000

        logger.info(
            "scoring_complete",
            transaction_id=request.transaction_id,
            action=action.value,
            fraud_probability=round(fraud_prob, 4),
            latency_ms=round(proc_time, 2),
        )

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
