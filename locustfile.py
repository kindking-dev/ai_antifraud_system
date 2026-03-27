import random
import uuid
from datetime import datetime, UTC
from locust import HttpUser, task, between


class RealWorldBankSimulator(HttpUser):
    # Увеличиваем паузу, чтобы обычные юзеры НЕ блокировались по скорости
    wait_time = between(10.0, 30.0)

    def on_start(self):
        # Огромный пул ID, чтобы имитировать базу большого города
        self.headers = {"X-API-KEY": "DEV-MASTER-KEY"}

    @task(90)  # 90% - НОРМАЛЬНЫЕ ЛЮДИ
    def clean_customer(self):
        user_id = f"USR-CLEAN-{random.randint(10000, 99999)}"
        payload = self._build_payload(user_id, is_fraud=False)
        self.client.post(
            "/v1/score-transaction",
            json=payload,
            headers=self.headers,
            name="01_Normal_Payment",
        )

    @task(7)  # 7% - КАРДИНГ (Всплеск частоты)
    def velocity_attacker(self):
        # Один и тот же юзер делает покупку каждые 0.5 сек
        user_id = "ATTACKER-VELOCITY-999"
        payload = self._build_payload(user_id, is_fraud=False)
        # У этой задачи НЕТ ожидания, она бьет быстро
        for _ in range(5):
            self.client.post(
                "/v1/score-transaction",
                json=payload,
                headers=self.headers,
                name="02_Velocity_Attack",
            )

    @task(3)  # 3% - БОТ-АТАКА (Аномальная биометрия)
    def biometric_bot(self):
        user_id = f"BOT-{uuid.uuid4().hex[:5]}"
        payload = self._build_payload(user_id, is_fraud=True)
        self.client.post(
            "/v1/score-transaction",
            json=payload,
            headers=self.headers,
            name="03_Bot_Anomalous_Biometrics",
        )

    def _build_payload(self, user_id, is_fraud):
        return {
            "transaction_id": f"TXN-{uuid.uuid4().hex[:8].upper()}",
            "user_id": user_id,
            "amount_kzt": random.uniform(200000, 1000000)
            if is_fraud
            else random.uniform(500, 50000),
            "source": "MOBILE_APP",
            "session_trust_score": 0.05 if is_fraud else 0.98,
            "network": {
                "ip_address": "1.1.1.1",
                "is_vpn_or_proxy": is_fraud,
                "ja3_fingerprint": "xyz",
                "user_agent": "Bot" if is_fraud else "iPhone",
            },
            "biometrics": {
                "gyroscope_x_y_z": [0.01, 0.01, 0.01] if is_fraud else [0.4, 0.7, 0.2],
                "keystroke_entropy": 0.1 if is_fraud else 0.8,
                "touch_pressure_variance": 0.02 if is_fraud else 0.18,
            },
            "timestamp_utc": datetime.now(UTC).isoformat(),
        }
