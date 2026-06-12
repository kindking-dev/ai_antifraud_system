"""
AI-POWERED BEHAVIORAL ANTI-FRAUD SERVICE: XAI Router.
Endpoint for generating Human-Readable Explanations via Local LLM.
"""

import logging
from fastapi import APIRouter, HTTPException, status

from app.schemas.llm import LLMExplanationRequest, LLMExplanationResponse
from app.services.llm_explainer import LLMExplainerService

logger = logging.getLogger(__name__)

# Инициализируем роутер с префиксом и тегом для Swagger UI
router = APIRouter(
    prefix="/explain",
    tags=["Explainable AI (XAI)"]
)

@router.post(
    "/transaction",
    response_model=LLMExplanationResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate AI Explanation for Transaction",
    description=(
        "Takes the results of the Late Fusion engine (CatBoost + HMOG) including SHAP values, "
        "and requests an asynchronous analysis from the local LLM (Qwen 2.5) to generate "
        "a human-readable markdown report for the security dashboard."
    )
)
async def explain_transaction(request_data: LLMExplanationRequest) -> LLMExplanationResponse:
    """
    Cold Path Endpoint: Вызывается аналитиком вручную из дашборда.
    Не влияет на SLA основного скоринга транзакций.
    """
    logger.info(f"Received XAI explanation request for TX: {request_data.transaction_id}")
    
    try:
        # Асинхронно вызываем наш сервис интеграции с LM Studio
        response = await LLMExplainerService.generate_explanation(request_data)
        
        # Логируем результат для метрик
        if response.status == "fallback":
            logger.warning(f"XAI generated fallback response for TX: {request_data.transaction_id}")
        else:
            logger.info(f"XAI generated successfully in {response.processing_time_ms}ms for TX: {request_data.transaction_id}")
            
        return response

    except Exception as e:
        logger.error(f"Critical failure in XAI router for TX {request_data.transaction_id}: {str(e)}")
        # Последний рубеж защиты: возвращаем 500, если упал сам сервис (хотя там есть свой fallback)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal Explainable AI System Error"
        )