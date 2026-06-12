import pytest
from datetime import datetime, timezone
from pydantic import ValidationError

from app.schemas.transaction import FraudAnalysisRequest, NetworkIdentity
from app.schemas.llm import LLMExplanationRequest

def test_network_identity_valid():
    network = NetworkIdentity(
        ip_address="192.168.1.1",
        ja3_fingerprint="a" * 32,  
        user_agent="Mozilla/5.0",
        is_vpn_or_proxy=True
    )
    # ИСПРАВЛЕНО: Приводим объект IPv4Address к строке для сравнения
    assert str(network.ip_address) == "192.168.1.1"
    assert network.is_vpn_or_proxy is True

def test_network_identity_invalid_ip():
    with pytest.raises(ValidationError):
        NetworkIdentity(
            ip_address="not-an-ip-address", 
            ja3_fingerprint="a" * 32,
            user_agent="Test",
            is_vpn_or_proxy=False
        )

def get_valid_fraud_payload() -> dict:
    return {
        "transaction_id": "TXN-123456",
        "user_id": "user_88",
        "amount_kzt": 15000.50,
        "source": "MOBILE_APP",
        "network": {
            "ip_address": "8.8.8.8",
            "ja3_fingerprint": "b" * 32,
            "user_agent": "iOS App",
            "is_vpn_or_proxy": False
        },
        "session_trust_score": 0.95,
        "timestamp_utc": datetime.now(timezone.utc).isoformat()
    }

def test_fraud_request_valid():
    payload = get_valid_fraud_payload()
    request = FraudAnalysisRequest(**payload)
    
    assert request.transaction_id == "TXN-123456"
    assert request.amount_kzt == 15000.50
    # ИСПРАВЛЕНО: Приводим объект к строке
    assert str(request.network.ip_address) == "8.8.8.8"

def test_fraud_request_negative_amount():
    payload = get_valid_fraud_payload()
    payload["amount_kzt"] = -500.0  
    
    with pytest.raises(ValidationError) as exc_info:
        FraudAnalysisRequest(**payload)
    assert "amount_kzt" in str(exc_info.value)

def test_fraud_request_invalid_trust_score():
    payload = get_valid_fraud_payload()
    payload["session_trust_score"] = 1.5
    with pytest.raises(ValidationError):
        FraudAnalysisRequest(**payload)
    payload["session_trust_score"] = -0.1
    with pytest.raises(ValidationError):
        FraudAnalysisRequest(**payload)

def test_llm_request_valid():
    request = LLMExplanationRequest(
        transaction_id="TXN-999",
        fraud_probability=0.88,
        action="BLOCK",
        feature_impacts={"velocity": 0.5, "vpn": 0.3},
        reason_codes=["HIGH_VELOCITY", "VPN_DETECTED"]
    )
    assert request.action == "BLOCK"
    assert len(request.feature_impacts) == 2