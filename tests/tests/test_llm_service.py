"""
SENTINEL AI: Unit tests for LLM Explainer Service.
Uses async mocking to simulate LM Studio behavior without requiring a real GPU or network connection.
"""

import pytest
import httpx
from unittest.mock import patch, AsyncMock, MagicMock

# Убедись, что пути импорта совпадают с твоей архитектурой
from app.schemas.llm import LLMExplanationRequest, LLMExplanationResponse
from app.services.llm_explainer import LLMExplainerService

@pytest.fixture
def mock_llm_request() -> LLMExplanationRequest:
    """Фикстура: типичный запрос на объяснение транзакции, который прилетел бы из Дашборда."""
    return LLMExplanationRequest(
        transaction_id="TXN-12345",
        fraud_probability=0.85,
        action="BLOCK",
        feature_impacts={"velocity": 0.45, "is_vpn": 0.3},
        reason_codes=["HIGH_VELOCITY", "VPN_DETECTED"]
    )


# ==========================================
# 1. ТЕСТ УСПЕШНОЙ ГЕНЕРАЦИИ (HAPPY PATH)
# ==========================================

@pytest.mark.asyncio
@patch("httpx.AsyncClient.post", new_callable=AsyncMock)
async def test_generate_explanation_success(mock_post, mock_llm_request):
    """
    Симуляция идеального ответа от Qwen 2.5 (LM Studio).
    """
    # 1. Настраиваем фейковый успешный HTTP-ответ
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": "**Transaction blocked** due to high velocity and VPN usage."
                }
            }
        ]
    }
    # Подменяем реальный запрос нашей заглушкой
    mock_post.return_value = mock_response

    # 2. Вызываем наш сервис
    response = await LLMExplainerService.generate_explanation(mock_llm_request)

    # 3. Проверяем результаты
    assert isinstance(response, LLMExplanationResponse)
    assert response.status == "success"
    assert "Transaction blocked" in response.explanation_markdown
    assert response.transaction_id == "TXN-12345"
    assert response.processing_time_ms > 0


# ==========================================
# 2. ТЕСТ ТАЙМАУТА (TIMEOUT DEGRADATION)
# ==========================================

@pytest.mark.asyncio
@patch("httpx.AsyncClient.post", new_callable=AsyncMock)
async def test_generate_explanation_timeout(mock_post, mock_llm_request):
    """
    Если нейросеть думает слишком долго, сервис должен вернуть fallback, а не зависнуть.
    """
    # Заставляем mock выбросить исключение Timeout
    mock_post.side_effect = httpx.TimeoutException("Connection timed out")

    response = await LLMExplainerService.generate_explanation(mock_llm_request)

    # Сервис должен элегантно обработать ошибку
    assert response.status == "fallback"
    assert "⚠️ **AI Assistant Unavailable.**" in response.explanation_markdown
    assert response.transaction_id == "TXN-12345"


# ==========================================
# 3. ТЕСТ ОФФЛАЙН СЕРВЕРА (CONNECTION ERROR)
# ==========================================

@pytest.mark.asyncio
@patch("httpx.AsyncClient.post", new_callable=AsyncMock)
async def test_generate_explanation_connection_error(mock_post, mock_llm_request):
    """
    Если LM Studio выключен (порт закрыт), сервис должен вернуть fallback.
    """
    # Для RequestError в httpx требуется передать объект request
    dummy_request = httpx.Request("POST", "http://localhost:1234/v1")
    mock_post.side_effect = httpx.RequestError("Connection refused", request=dummy_request)

    response = await LLMExplainerService.generate_explanation(mock_llm_request)

    # Сервис должен элегантно обработать отвал сети
    assert response.status == "fallback"
    assert "⚠️ **AI Assistant Unavailable.**" in response.explanation_markdown
    assert response.transaction_id == "TXN-12345"