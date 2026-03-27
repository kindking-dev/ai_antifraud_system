import random
import uuid
import hashlib
from datetime import datetime, timezone
from locust import HttpUser, task, between


class AntiFraudRealWorldSimulator(HttpUser):
    # Имитируем реальное время раздумья человека (от 10 до 20 секунд)
    wait_time = between(10.0, 20.0)

    def on_start(self):
        """Вызывается при старте каждого виртуального юзера."""
        self.headers = {
            "Content-Type": "application/json",
            "X-API-KEY": "DEV-MASTER-KEY",
        }

    @task(85)  # 85% - Обычные клиенты (Нормальное поведение)
    def normal_customer(self):
        user_id = f"USR-{random.randint(10000, 99999)}"
        payload = self._build_payload(user_id, profile="CLEAN")
        self.client.post(
            "/v1/score-transaction",
            json=payload,
            headers=self.headers,
            name="01_Normal_User",
        )

    @task(10)  # 10% - Атака на скорость (Velocity / Carding)
    def velocity_attacker(self):
        # Один и тот же злоумышленник "долбит" систему 5 раз подряд
        attacker_id = "ATTACKER-VELOCITY-001"
        for _ in range(5):
            payload = self._build_payload(attacker_id, profile="VELOCITY")
            self.client.post(
                "/v1/score-transaction",
                json=payload,
                headers=self.headers,
                name="02_Velocity_Attack",
            )

    @task(5)  # 5% - Продвинутый фрод (Аномальная биометрия)
    def advanced_fraud_bot(self):
        user_id = f"BOT-{uuid.uuid4().hex[:5].upper()}"
        payload = self._build_payload(user_id, profile="FRAUD_BOT")
        self.client.post(
            "/v1/score-transaction",
            json=payload,
            headers=self.headers,
            name="03_Biometric_Bot_Attack",
        )

    def _build_payload(self, user_id, profile):
        # 1. Генерируем JA3 хеш ровно 32 символа (требование твоей модели)
        ja3 = hashlib.md5(user_id.encode()).hexdigest()

        # 2. Логика в зависимости от профиля
        is_fraud = profile in ["VELOCITY", "FRAUD_BOT"]

        # Для ботов биометрия "мёртвая" (нули), для людей - живая
        if profile == "FRAUD_BOT":
            biometrics = {
                "gyroscope_x_y_z": [0.0, 0.0, 0.0],
                "keystroke_entropy": 0.01,
                "touch_pressure_variance": 0.001,
            }
            amount = float(random.randint(500000, 1500000))
            trust = 0.05
        else:
            biometrics = {
                "gyroscope_x_y_z": [
                    round(random.uniform(0.1, 0.8), 2) for _ in range(3)
                ],
                "keystroke_entropy": round(random.uniform(0.6, 0.9), 2),
                "touch_pressure_variance": round(random.uniform(0.1, 0.3), 2),
            }
            amount = float(random.randint(500, 45000))
            trust = 0.98

        # Собираем финальный JSON строго по твоей схеме Pydantic
        return {
            "transaction_id": f"TXN-{uuid.uuid4().hex[:15].upper()}",
            "user_id": user_id,
            "amount_kzt": amount,
            "source": "MOBILE_APP" if profile != "FRAUD_BOT" else "API",
            "network": {
                "ip_address": f"{random.randint(1, 255)}.{random.randint(1, 255)}.1.1",
                "ja3_fingerprint": ja3,
                "user_agent": "iPhone/Safari"
                if not is_fraud
                else "Python/Requests-Bot",
                "is_vpn_or_proxy": is_fraud,
            },
            "biometrics": biometrics,
            "session_trust_score": trust,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
