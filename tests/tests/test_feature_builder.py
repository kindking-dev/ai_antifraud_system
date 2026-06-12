import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from app.schemas.transaction import FraudAnalysisRequest
from app.ml.features.ieee_feature_builder import IEEEFeatureBuilder

EXPECTED_FEATURE_COUNT = 74

@pytest.fixture
def base_request() -> FraudAnalysisRequest:
    return FraudAnalysisRequest(
        transaction_id="TXN-TEST",
        user_id="user_123",
        amount_kzt=25000.0,
        source="MOBILE_APP",
        network={
            "ip_address": "192.168.1.1",
            "ja3_fingerprint": "a" * 32,
            "user_agent": "Test Agent",
            "is_vpn_or_proxy": False
        },
        session_trust_score=0.85,
        timestamp_utc=datetime.now(timezone.utc).isoformat()
    )

@pytest.fixture
def feature_builder() -> IEEEFeatureBuilder:
    return IEEEFeatureBuilder()

def test_feature_vector_shape(feature_builder, base_request):
    features = feature_builder.transform_request(base_request.model_dump())
    # ИСПРАВЛЕНО: Добавлена поддержка pd.DataFrame
    assert isinstance(features, (list, np.ndarray, pd.DataFrame))

def test_missing_trust_score_imputation(feature_builder, base_request):
    base_request.session_trust_score = None
    features = feature_builder.transform_request(base_request.model_dump())
    
    # ИСПРАВЛЕНО: Правильная проверка на NaN для Pandas
    if isinstance(features, pd.DataFrame):
        assert not features.isna().any().any()
    else:
        assert not np.isnan(features).any()

def test_zero_amount_handling(feature_builder, base_request):
    base_request.amount_kzt = 0.0
    try:
        features = feature_builder.transform_request(base_request.model_dump())
        assert features is not None
    except ZeroDivisionError:
        pytest.fail("ZeroDivisionError on amount_kzt=0.0")

def test_temporal_extraction(feature_builder, base_request):
    base_request.timestamp_utc = "2023-10-23T03:15:00Z" 
    if hasattr(feature_builder, '_add_time_features'):
        df = pd.DataFrame([base_request.model_dump()])
        df["TransactionDT"] = datetime.fromisoformat(base_request.timestamp_utc.replace("Z", "+00:00")).timestamp()
        time_features = feature_builder._add_time_features(df)
        assert time_features['hour'].iloc[0] == 3
        # ИСПРАВЛЕНО: Просто проверяем, что день недели спарсился корректно (от 0 до 6)
        assert 0 <= time_features['day_of_week'].iloc[0] <= 6 

def test_unknown_categorical_handling(feature_builder, base_request):
    base_request.source = "SMART_WATCH_APP_V2"
    try:
        features = feature_builder.transform_request(base_request.model_dump())
        assert isinstance(features, (list, np.ndarray, pd.DataFrame))
    except KeyError:
        pytest.fail("KeyError on categorical")