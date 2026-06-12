"""
SENTINEL AI: System Integration & Late Fusion Tester.
Full Compliance Edition (Satisfies strict biometric schemas).
"""

import asyncio
import httpx
import json
import uuid
from datetime import datetime, timezone
from typing import Dict, Any

# --- CONFIGURATION ---
BASE_URL = "http://127.0.0.1:8000/v1"
HEADERS = {"X-API-KEY": "DEV-MASTER-KEY"}
TEST_USER_ID = "7" 
TIMEOUT = 12.0

class Color:
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    END = '\033[0m'

async def log_step(message: str, color=Color.BLUE):
    print(f"{color}{message}{Color.END}")

def get_full_features(velocity: float, pressure: float, area: float) -> Dict[str, float]:
    """Генерирует полный набор признаков для соответствия Pydantic-схеме."""
    return {
        "duration_ms_mean": 250.0, "duration_ms_std": 30.0, "duration_ms_max": 350.0,
        "length_px_mean": 480.0, "length_px_std": 50.0, "length_px_max": 600.0,
        "velocity_mean": velocity, "velocity_std": velocity * 0.1, "velocity_max": velocity * 1.2,
        "median_pressure_mean": pressure, "median_pressure_std": 0.05, "median_pressure_max": pressure + 0.1,
        "median_area_mean": area, "median_area_std": 0.05, "median_area_max": area + 0.1
    }

async def run_scenario(client: httpx.AsyncClient, name: str, behavior_data: Dict, tx_data: Dict) -> Dict:
    results = {"scenario": name, "timestamp": datetime.now().isoformat()}
    await log_step(f"\n🚀 SCENARIO: {name}")
    
    # 1. Behavioral Scoring
    try:
        resp_b = await client.post(f"{BASE_URL}/score-behavior", json=behavior_data, headers=HEADERS)
        results["behavior"] = resp_b.json()
        if resp_b.status_code == 200:
            print(f"  ✅ Bio: Risk {results['behavior']['fraud_probability']} | {results['behavior']['status']}")
        else:
            print(f"  ❌ Bio Error {resp_b.status_code}: {resp_b.text}")
    except Exception as e:
        print(f"  🔥 Bio Connection Failed: {e}")

    await asyncio.sleep(0.6) # Пауза для Redis

    # 2. Transaction Scoring
    try:
        resp_t = await client.post(f"{BASE_URL}/score-transaction", json=tx_data, headers=HEADERS)
        results["transaction"] = resp_t.json()
        if resp_t.status_code == 200:
            print(f"  ✅ Tx: {Color.BOLD}{results['transaction']['action']}{Color.END} | Prob: {results['transaction']['fraud_probability']}")
        else:
            print(f"  ❌ Tx Error {resp_t.status_code}: {resp_t.text}")
    except Exception as e:
        print(f"  🔥 Tx Connection Failed: {e}")

    return results

async def main():
    report = []
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        # 1. LEGIT
        report.append(await run_scenario(
            client, "LEGITIMATE_OWNER",
            {"user_id": TEST_USER_ID, "features": get_full_features(1.2, 0.15, 0.3)},
            {
                "transaction_id": f"TXN-LEGIT-{uuid.uuid4().hex[:4].upper()}",
                "user_id": TEST_USER_ID, "amount_kzt": 5000.0, "source": "MOBILE_APP",
                "session_trust_score": 0.98,
                "network": {"ip_address": "127.0.0.1", "ja3_fingerprint": "a"*32, "user_agent": "iOS", "is_vpn_or_proxy": False}
            }
        ))

        # 2. ATTACK
        report.append(await run_scenario(
            client, "BEHAVIORAL_ANOMALY_ATTACK",
            {"user_id": TEST_USER_ID, "features": get_full_features(15.5, 0.95, 0.85)},
            {
                "transaction_id": f"TXN-ATTACK-{uuid.uuid4().hex[:4].upper()}",
                "user_id": TEST_USER_ID, "amount_kzt": 15000.0, "source": "MOBILE_APP",
                "session_trust_score": 0.4,
                "network": {"ip_address": "8.8.8.8", "ja3_fingerprint": "b"*32, "user_agent": "Bot", "is_vpn_or_proxy": True}
            }
        ))

    # Фикс json.dump
    with open("test_results.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, ensure_ascii=False)
    
    print(f"\n{Color.GREEN}{Color.BOLD}SUCCESS!{Color.END} Results saved to: test_results.json")

if __name__ == "__main__":
    asyncio.run(main())