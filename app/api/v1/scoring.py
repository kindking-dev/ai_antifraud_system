from fastapi import APIRouter, HTTPException, Depends
import structlog
import time

from app.schemas.transaction import FraudAnalysisRequest
from app.schemas.response import FraudAnalysisResponse, FraudAction, ReasonCode

logger = structlog.get_logger()

# ВОТ ТОТ САМЫЙ ОБЪЕКТ ROUTER, КОТОРЫЙ ИСКАЛ MAIN.PY
router = APIRouter()


async def get_db_session():
    """Заглушка для будущей базы данных"""
    yield "db_session_mock"


@router.post(
    "/score-transaction",
    response_model=FraudAnalysisResponse,
    summary="Анализ транзакции на мошенничество",
)
async def score_transaction(request: FraudAnalysisRequest, db=Depends(get_db_session)):
    start_time = time.perf_counter()
    logger.info("Processing transaction", transaction_id=request.transaction_id)

    try:
        # Временная мок-логика для проверки
        fraud_prob = 0.05
        action = FraudAction.ALLOW
        reasons = []

        if request.session_trust_score < 0.3:
            fraud_prob += 0.4
            action = FraudAction.CHALLENGE
            reasons.append(ReasonCode.TRUST_DECAY)

        if request.network.is_vpn_or_proxy:
            fraud_prob += 0.3
            action = FraudAction.BLOCK if fraud_prob > 0.8 else FraudAction.CHALLENGE
            reasons.append(ReasonCode.NETWORK_RISK)

        processing_time = (time.perf_counter() - start_time) * 1000

        return FraudAnalysisResponse(
            transaction_id=request.transaction_id,
            action=action,
            fraud_probability=min(fraud_prob, 1.0),
            reason_codes=reasons,
            processing_time_ms=round(processing_time, 2),
        )

    except Exception as e:
        logger.error("Scoring failed", error=str(e))
        raise HTTPException(status_code=500, detail="Internal ML engine error")
