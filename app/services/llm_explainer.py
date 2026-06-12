"""
AI-POWERED BEHAVIORAL ANTI-FRAUD SERVICE: LLM Explainer Service.
Asynchronous integration with local LM Studio (Qwen 2.5) for XAI reporting.
Supports Docker environments via environment variables.
"""

import logging
import time
import httpx
import os
from typing import Dict, Any

from app.schemas.llm import LLMExplanationRequest, LLMExplanationResponse

logger = logging.getLogger(__name__)

# Динамическая конфигурация для Docker (Fallback на localhost для локального дева)
LLM_API_URL = os.getenv("LLM_API_URL", "http://localhost:1234/v1/chat/completions")
MODEL_NAME = "qwen2.5-7b-instruct"
TIMEOUT_SEC = 15.0  # Увеличен таймаут для работы через сетевой мост Docker

class LLMExplainerService:
    """
    Сервис-оркестратор для связи FastAPI с локальной LLM.
    Реализует паттерн Cold Path (не блокирует основной инференс антифрода).
    """

    @staticmethod
    async def generate_explanation(request_data: LLMExplanationRequest) -> LLMExplanationResponse:
        """
        Отправляет контекст транзакции и SHAP-веса в нейросеть для получения человекочитаемого отчета.
        """
        start_time = time.perf_counter()

        # 1. Системный промпт (Защита от галлюцинаций и установка роли)
        system_prompt = (
            "You are a Senior Anti-Fraud Analyst at a top-tier fintech company. "
            "Your task is to explain why a transaction was blocked, challenged, or allowed "
            "based STRICTLY on CatBoost ML probabilities, Reason Codes, and SHAP feature impacts. "
            "Rules: "
            "1. Be concise, highly professional, and write under 4 sentences. "
            "2. Use markdown formatting (bolding for key metrics and risk factors). "
            "3. Do NOT hallucinate data or invent new features. Only use the provided context."
        )

        # 2. Пользовательский промпт (Математика и факты)
        # Преобразуем SHAP веса в читаемый вид (сортируем по модулю влияния)
        sorted_impacts = sorted(
            request_data.feature_impacts.items(), 
            key=lambda item: abs(item[1]), 
            reverse=True
        )[:5] # Берем топ-5 самых важных фичей
        
        impacts_str = ", ".join([f"{k}: {v:.4f}" for k, v in sorted_impacts])
        reason_codes_str = ", ".join(request_data.reason_codes) if request_data.reason_codes else "None"

        user_prompt = (
            f"Transaction ID: {request_data.transaction_id}\n"
            f"System Verdict: {request_data.action}\n"
            f"Fraud Probability: {request_data.fraud_probability:.4f}\n"
            f"Triggered Reason Codes: {reason_codes_str}\n"
            f"Top SHAP Feature Impacts: {impacts_str}\n\n"
            "Explain the primary drivers for this verdict based ONLY on the provided data. "
            "If behavioral anomalies (HMOG) are present, explicitly state that user biometrics do not match the historical profile."
        )

        # 3. Формирование Payload (OpenAI format)
        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.1,  # Минимальная креативность = максимальная фактологичность
            "max_tokens": 200    # Ограничение размера ответа
        }

        try:
            # 4. Асинхронный HTTP запрос
            async with httpx.AsyncClient(timeout=TIMEOUT_SEC) as client:
                response = await client.post(LLM_API_URL, json=payload)
                response.raise_for_status()

                result = response.json()
                explanation = result["choices"][0]["message"]["content"]

                latency = (time.perf_counter() - start_time) * 1000

                return LLMExplanationResponse(
                    transaction_id=request_data.transaction_id,
                    explanation_markdown=explanation.strip(),
                    processing_time_ms=round(latency, 2),
                    status="success"
                )

        except httpx.TimeoutException:
            logger.error(f"[LLM Timeout] URL {LLM_API_URL} did not respond in time for TX {request_data.transaction_id}")
            latency = (time.perf_counter() - start_time) * 1000
            return LLMExplainerService._fallback_response(request_data.transaction_id, latency)
            
        except httpx.RequestError as e:
            logger.error(f"[LLM Connection Error] LM Studio offline at {LLM_API_URL}. TX {request_data.transaction_id}: {e}")
            latency = (time.perf_counter() - start_time) * 1000
            return LLMExplainerService._fallback_response(request_data.transaction_id, latency)
            
        except Exception as e:
            logger.error(f"[LLM Generation Failed] Unexpected error for TX {request_data.transaction_id}: {e}")
            latency = (time.perf_counter() - start_time) * 1000
            return LLMExplainerService._fallback_response(request_data.transaction_id, latency)

    @staticmethod
    def _fallback_response(tx_id: str, latency: float) -> LLMExplanationResponse:
        """
        Graceful Degradation: возвращает безопасную заглушку, 
        если AI-ассистент недоступен или изолирован в Docker-сети.
        """
        fallback_msg = (
            "⚠️ **AI Assistant Unavailable.**\n"
            "The localized LLM engine is currently offline or disconnected from the Docker network. "
            "Please review the SHAP charts manually in the dashboard to determine the anomaly root cause."
        )
        return LLMExplanationResponse(
            transaction_id=tx_id,
            explanation_markdown=fallback_msg,
            processing_time_ms=round(latency, 2),
            status="fallback"
        )