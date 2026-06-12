"""
SENTINEL AI: ML Model Validator & Quality Assurance Script.
Generates synthetic diverse traffic to evaluate False Positives, 
True Positives, and SLA latency of the Late Fusion engine.
"""

import httpx
import random
import uuid
import time
import pandas as pd
from datetime import datetime, timezone
from typing import List, Dict, Any

from app.core.config import settings

# Конфигурация API
BASE_URL = f"http://localhost:8000{settings.API_V1_STR}"
API_KEY = "DEV-MASTER-KEY"
HEADERS = {"X-API-KEY": API_KEY, "Content-Type": "application/json"}

# Настройки теста
TOTAL_USERS = 100  # Сколько всего симуляций провести
FRAUD_RATIO = 0.3  # 30% фрода, 70% честных юзеров (реалистичный стресс-сценарий)

class DataGenerator:
    """Генератор синтетических поведенческих и транзакционных профилей."""
    
    @staticmethod
    def generate_normal_user() -> Dict[str, Any]:
        return {
            "type": "NORMAL",
            "expected_action": "ALLOW",
            "trust": random.uniform(0.7, 1.0),
            "velocity": random.uniform(0.5, 3.0),
            "pressure": random.uniform(0.3, 0.6),
            "vpn": False,
            "amount": round(random.uniform(1000, 50000), 2)
        }

    @staticmethod
    def generate_account_takeover() -> Dict[str, Any]:
        return {
            "type": "ATO_FRAUD",
            "expected_action": "BLOCK",
            "trust": random.uniform(0.1, 0.4),
            "velocity": random.uniform(6.0, 12.0), # Нервные, быстрые движения
            "pressure": random.uniform(0.8, 1.0),  # Сильное давление
            "vpn": random.choice([True, False]),
            "amount": round(random.uniform(100000, 500000), 2)
        }

    @staticmethod
    def generate_bot_script() -> Dict[str, Any]:
        return {
            "type": "BOT_FRAUD",
            "expected_action": "BLOCK",
            "trust": random.uniform(0.0, 0.1),
            "velocity": random.uniform(15.0, 25.0), # Нечеловеческая скорость
            "pressure": random.uniform(0.0, 0.05),  # Идеально ровное/нулевое давление
            "vpn": True,
            "amount": round(random.uniform(10000, 200000), 2)
        }

def run_ml_validation_suite():
    print("🚀 Initiating Sentinel AI Validation Suite...")
    print(f"📊 Target Volume: {TOTAL_USERS} Transactions (Fraud Ratio: {FRAUD_RATIO*100}%)")
    
    results = []
    
    with httpx.Client(timeout=10.0) as client:
        for i in range(TOTAL_USERS):
            user_id = f"val_usr_{uuid.uuid4().hex[:6]}"
            
            # 1. Выбираем тип юзера
            if random.random() < FRAUD_RATIO:
                profile = DataGenerator.generate_bot_script() if random.random() < 0.5 else DataGenerator.generate_account_takeover()
            else:
                profile = DataGenerator.generate_normal_user()

            # 2. Шаг 1: Стриминг биометрии (HMOG)
            behavior_payload = {
                "user_id": user_id,
                "features": {
                    "duration_ms_mean": 250.0, "duration_ms_std": 20.0, "duration_ms_max": 300.0,
                    "length_px_mean": 450.0, "length_px_std": 30.0, "length_px_max": 500.0,
                    "velocity_mean": profile["velocity"], "velocity_std": 1.0, "velocity_max": profile["velocity"] + 2,
                    "median_pressure_mean": profile["pressure"], "median_pressure_std": 0.05, "median_pressure_max": profile["pressure"] + 0.1,
                    "median_area_mean": 0.5, "median_area_std": 0.05, "median_area_max": 0.6
                }
            }
            client.post(f"{BASE_URL}/score-behavior", json=behavior_payload, headers=HEADERS)

            # 3. Шаг 2: Транзакция (CatBoost)
            tx_payload = {
                "transaction_id": f"TXN-{uuid.uuid4().hex[:8].upper()}",
                "user_id": user_id,
                "amount_kzt": profile["amount"],
                "source": "MOBILE_APP",
                "network": {
                    "ip_address": "192.168.1.1",
                    "ja3_fingerprint": "a" * 32,
                    "user_agent": "Validation_Suite",
                    "is_vpn_or_proxy": profile["vpn"]
                },
                "session_trust_score": profile["trust"],
                "timestamp_utc": datetime.now(timezone.utc).isoformat()
            }
            
            resp = client.post(f"{BASE_URL}/score-transaction", json=tx_payload, headers=HEADERS)
            
            if resp.status_code == 200:
                data = resp.json()
                results.append({
                    "tx_id": data["transaction_id"],
                    "profile_type": profile["type"],
                    "expected_action": profile["expected_action"],
                    "actual_action": data["action"],
                    "fraud_prob": data["fraud_probability"],
                    "latency_ms": data["processing_time_ms"]
                })
                print(f"Processed {i+1}/{TOTAL_USERS} | Type: {profile['type']:<10} | Result: {data['action']:<10} | Latency: {data['processing_time_ms']}ms")
            else:
                print(f"❌ Error on {i+1}: {resp.text}")

    # =========================================
    # ML ANALYTICS REPORT
    # =========================================
    df = pd.DataFrame(results)
    
    total_normal = len(df[df['expected_action'] == 'ALLOW'])
    total_fraud = len(df[df['expected_action'] == 'BLOCK'])
    
    # False Positives: Блочит всех подряд (Ожидали ALLOW, получили BLOCK/CHALLENGE)
    false_positives = len(df[(df['expected_action'] == 'ALLOW') & (df['actual_action'] != 'ALLOW')])
    fp_rate = (false_positives / total_normal * 100) if total_normal > 0 else 0.0
    
    # True Positives: Успешно поймали фрод (Ожидали BLOCK, получили BLOCK/CHALLENGE)
    true_positives = len(df[(df['expected_action'] == 'BLOCK') & (df['actual_action'] != 'ALLOW')])
    tp_rate = (true_positives / total_fraud * 100) if total_fraud > 0 else 0.0

    avg_latency = df['latency_ms'].mean()
    p95_latency = df['latency_ms'].quantile(0.95)

    print("\n" + "="*50)
    print("🏆 SENTINEL AI: ML MODEL VALIDATION REPORT 🏆")
    print("="*50)
    print(f"Total Transactions Analyzed: {TOTAL_USERS}")
    print(f"Normal Flow (Legit Users)  : {total_normal}")
    print(f"Abnormal Flow (Fraudsters) : {total_fraud}")
    print("-" * 50)
    print("🚨 ML PERFORMANCE METRICS:")
    # Главная метрика для комиссии - низкий FP Rate
    print(f"False Positive Rate (Blocked Legit Users): {fp_rate:.2f}% " + ("(✅ EXCELLENT)" if fp_rate < 5 else "(⚠️ HIGH)"))
    print(f"True Positive Rate (Caught Fraudsters)   : {tp_rate:.2f}% " + ("(✅ EXCELLENT)" if tp_rate > 90 else "(⚠️ LOW)"))
    print("-" * 50)
    print("⚡ SYSTEM SLA METRICS:")
    print(f"Average Latency : {avg_latency:.2f} ms")
    print(f"P95 Latency     : {p95_latency:.2f} ms (SLA < 50ms: {'✅ PASS' if p95_latency < 50 else '❌ FAIL'})")
    print("="*50)

if __name__ == "__main__":
    run_ml_validation_suite()