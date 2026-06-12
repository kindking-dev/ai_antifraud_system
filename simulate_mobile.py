import requests
import json
import time

headers = {"X-API-KEY": "DEV-MASTER-KEY", "Content-Type": "application/json"}
uid = "live_mobile_user_01"

# 1. SCORE BEHAVIOR
behavior_payload = {
    "user_id": uid,
    "features": {
        "duration_ms_mean": 250.0,
        "duration_ms_std": 10.0,
        "duration_ms_max": 300.0,
        "length_px_mean": 400.0,
        "length_px_std": 15.0,
        "length_px_max": 450.0,
        "velocity_mean": 1.5,
        "velocity_std": 0.2,
        "velocity_max": 2.0,
        "median_pressure_mean": 0.5,
        "median_pressure_std": 0.05,
        "median_pressure_max": 0.6,
        "median_area_mean": 5.0,
        "median_area_std": 0.5,
        "median_area_max": 6.0
    }
}
print("--- BEHAVIOR ---")
resp = requests.post("http://localhost:8000/api/v1/score-behavior", json=behavior_payload, headers=headers)
print(resp.json())

time.sleep(1)

# 2. SCORE TRANSACTION
tx_payload = {
    "transaction_id": "TXN-12345",
    "user_id": uid,
    "amount_kzt": 15000.0,
    "source": "MOBILE_APP",
    "session_trust_score": 0.99,
    "network": {
        "ip_address": "192.168.1.15", 
        "ja3_fingerprint": "a"*32,
        "user_agent": "Mobile",
        "is_vpn_or_proxy": False 
    },
    "biometrics": {
        **behavior_payload["features"],
        "gyroscope_x_y_z": [1.0, 2.0, 3.0],
        "keystroke_entropy": 0.1,
        "touch_pressure_variance": 0.01
    }
}

print("--- TRANSACTION ---")
resp = requests.post("http://localhost:8000/api/v1/score-transaction", json=tx_payload, headers=headers)
print(resp.json())
