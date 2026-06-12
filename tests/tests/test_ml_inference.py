"""
SENTINEL AI: ML Inference & MLOps Tests.
Validates CatBoost model initialization, prediction latency (SLA < 50ms), 
determinism, SHAP generation, and adversarial input handling.
"""

import os
import time
import tempfile
import pytest
import numpy as np
from typing import Dict, Any
from catboost import CatBoostClassifier, Pool

# ВАЖНО: Замени пути импорта на свои реальные классы.
# Предполагается, что FraudService инкапсулирует загрузку модели и инференс.
# from app.services.fraud_service import FraudService 

EXPECTED_FEATURE_COUNT = 74
SLA_MAX_LATENCY_MS = 50.0

# ==========================================
# ФИКСТУРЫ (ML SETUP)
# ==========================================

@pytest.fixture(scope="session")
def dummy_catboost_model_path():
    """
    Создает крошечную dummy-модель CatBoost для CI/CD тестов.
    Гарантирует, что тесты пройдут даже без скачивания боевых весов.
    """
    np.random.seed(42)
    # Генерируем 10 случайных сэмплов с 74 фичами
    X_dummy = np.random.rand(10, EXPECTED_FEATURE_COUNT)
    y_dummy = np.random.randint(0, 2, 10)
    
    model = CatBoostClassifier(
        iterations=5, 
        depth=2, 
        learning_rate=0.1, 
        loss_function='Logloss',
        verbose=False
    )
    model.fit(X_dummy, y_dummy)
    
    # Сохраняем во временный файл, который удалится после тестов
    fd, path = tempfile.mkstemp(suffix=".cbm")
    os.close(fd)
    model.save_model(path)
    
    yield path
    
    # Очистка
    os.remove(path)


@pytest.fixture
def mock_fraud_service(dummy_catboost_model_path: str):
    """
    Фикстура, имитирующая инициализацию сервиса инференса с тестовой моделью.
    В реальном коде замени это на инициализацию твоего класса.
    """
    class MockFraudService:
        def __init__(self, model_path: str):
            self.model = CatBoostClassifier()
            self.model.load_model(model_path)
            
        def predict(self, feature_vector: np.ndarray) -> Dict[str, Any]:
            start_time = time.perf_counter()
            
            # Защита размерности
            if feature_vector.shape[0] != EXPECTED_FEATURE_COUNT:
                raise ValueError(f"Feature shape mismatch: {feature_vector.shape}")
                
            # Защита от NaN / Inf (замена на 0.0)
            clean_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)
            
            prob = self.model.predict_proba(clean_vector)[1]
            
            # Симуляция SHAP (в реальном коде вызывается TreeExplainer)
            shap_values = np.random.uniform(-0.1, 0.1, EXPECTED_FEATURE_COUNT)
            
            latency = (time.perf_counter() - start_time) * 1000
            
            return {
                "fraud_probability": float(prob),
                "action": "BLOCK" if prob > 0.8 else ("CHALLENGE" if prob > 0.4 else "ALLOW"),
                "processing_time_ms": latency,
                "shap_values": shap_values.tolist()
            }
            
    return MockFraudService(dummy_catboost_model_path)


@pytest.fixture
def normal_feature_vector() -> np.ndarray:
    """Нормальный вектор фичей легитимного юзера."""
    np.random.seed(100)
    return np.random.rand(EXPECTED_FEATURE_COUNT).astype(np.float32)


# ==========================================
# ТЕСТЫ ИНФЕРЕНСА
# ==========================================

def test_inference_latency_sla(mock_fraud_service, normal_feature_vector):
    """
    КРИТИЧЕСКИЙ ТЕСТ: Инференс одной транзакции должен укладываться в SLA < 50ms.
    """
    # Прогрев (Warm-up)
    _ = mock_fraud_service.predict(normal_feature_vector)
    
    latencies = []
    for _ in range(100):
        result = mock_fraud_service.predict(normal_feature_vector)
        latencies.append(result["processing_time_ms"])
        
    avg_latency = sum(latencies) / len(latencies)
    p95_latency = np.percentile(latencies, 95)
    
    assert p95_latency < SLA_MAX_LATENCY_MS, f"SLA Violation! P95 Latency is {p95_latency:.2f}ms"
    assert avg_latency > 0.0


def test_determinism_no_data_leakage(mock_fraud_service, normal_feature_vector):
    """
    Проверка детерминированности: одинаковый вход должен давать строго одинаковый выход.
    Защищает от багов состояния (State Leakage) между запросами.
    """
    result_1 = mock_fraud_service.predict(normal_feature_vector)
    result_2 = mock_fraud_service.predict(normal_feature_vector)
    
    assert result_1["fraud_probability"] == pytest.approx(result_2["fraud_probability"], rel=1e-6)
    assert result_1["action"] == result_2["action"]


def test_adversarial_nan_and_inf_handling(mock_fraud_service):
    """
    Adversarial Test: Модель не должна падать с C++ Exception, 
    если в нее прилетит вектор из NaN или Infinity.
    """
    poisoned_vector = np.full(EXPECTED_FEATURE_COUNT, np.nan, dtype=np.float32)
    poisoned_vector[5] = np.inf
    
    try:
        result = mock_fraud_service.predict(poisoned_vector)
        assert isinstance(result["fraud_probability"], float)
        assert not np.isnan(result["fraud_probability"])
    except Exception as e:
        pytest.fail(f"ML Service crashed on poisoned NaN/Inf vector! Exception: {e}")


def test_shap_values_generation(mock_fraud_service, normal_feature_vector):
    """
    Explainability (XAI) Test: Проверка генерации SHAP весов для LLM.
    Количество весов должно строго совпадать с количеством фичей.
    """
    result = mock_fraud_service.predict(normal_feature_vector)
    
    assert "shap_values" in result
    shap_vals = result["shap_values"]
    
    assert len(shap_vals) == EXPECTED_FEATURE_COUNT, \
        f"SHAP weights count ({len(shap_vals)}) mismatch feature count ({EXPECTED_FEATURE_COUNT})"
    assert all(isinstance(val, float) for val in shap_vals)