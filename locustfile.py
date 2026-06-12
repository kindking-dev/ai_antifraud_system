import os
import random
import uuid
from datetime import datetime, timezone

from locust import HttpUser, between, task


API_KEY = os.getenv("API_KEY", "DEV-MASTER-KEY")
USER_POOL_SIZE = int(os.getenv("LOCUST_USER_POOL_SIZE", "10000"))

# core: production hot path only. lifecycle: includes /set-baseline startup calls.
LOCUST_PROFILE = os.getenv("LOCUST_PROFILE", "core").strip().lower()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def behavior_payload(user_id: str) -> dict:
    base_x = random.randint(80, 180)
    base_y = random.randint(180, 420)
    started_at = random.randint(1_000_000, 9_000_000)

    events = []
    for i in range(15):
        x = base_x + random.randint(-12, 12) + i * random.randint(2, 8)
        y = base_y + random.randint(-18, 18) + i * random.randint(1, 6)
        events.append(
            {
                # Keep both naming styles so load tests match current demos and engine code.
                "x": x,
                "y": y,
                "x_pos": x,
                "y_pos": y,
                "timestamp_ms": started_at + i * random.randint(18, 45),
                "pressure": round(random.uniform(0.35, 0.85), 3),
                "contact_size": round(random.uniform(0.04, 0.18), 3),
            }
        )

    return {
        "user_id": user_id,
        "device_id": f"android-{random.randint(1, 500)}",
        "events": events,
        "timestamp": now_iso(),
    }


def transaction_payload(user_id: str) -> dict:
    amount = round(random.choice([1500, 3200, 7500, 15000, 45000, 120000]) * random.uniform(0.8, 1.25), 2)
    suspicious = random.random() < 0.08

    return {
        "transaction_id": f"tx-{uuid.uuid4()}",
        "user_id": user_id,
        "amount_kzt": amount,
        "source": random.choice(["MOBILE_APP", "WEB", "API"]),
        "session_trust_score": round(random.uniform(0.15, 0.45) if suspicious else random.uniform(0.72, 0.98), 3),
        "network": {
            "ip_address": f"10.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 254)}",
            "ja3_fingerprint": uuid.uuid4().hex,
            "user_agent": random.choice(
                [
                    "SentinelMobile/1.0 Android",
                    "SentinelMobile/1.0 iOS",
                    "Mozilla/5.0 Chrome/126.0",
                ]
            ),
            "is_vpn_or_proxy": suspicious,
        },
        "biometrics": {
            "gyroscope_x_y_z": [
                round(random.uniform(-0.8, 0.8), 3),
                round(random.uniform(-0.8, 0.8), 3),
                round(random.uniform(8.8, 10.2), 3),
            ],
            "keystroke_entropy": round(random.uniform(0.2, 1.8), 3),
            "touch_pressure_variance": round(random.uniform(0.01, 0.22), 3),
        },
        "timestamp_utc": now_iso(),
    }


class AntifraudUser(HttpUser):
    wait_time = between(0.2, 1.2)

    def on_start(self):
        self.headers = {
            "Content-Type": "application/json",
            "X-API-KEY": API_KEY,
        }
        self.user_id = f"user-{random.randint(1, USER_POOL_SIZE)}"

        if LOCUST_PROFILE == "lifecycle":
            self.client.post(
                "/api/v1/set-baseline",
                json=behavior_payload(self.user_id),
                headers=self.headers,
                name="/api/v1/set-baseline",
            )

    @task(1)
    def health(self):
        with self.client.get("/health", name="/health", catch_response=True) as response:
            if response.status_code != 200:
                response.failure(f"unexpected status {response.status_code}")

    @task(3)
    def score_behavior(self):
        with self.client.post(
            "/api/v1/score-behavior",
            json=behavior_payload(self.user_id),
            headers=self.headers,
            name="/api/v1/score-behavior",
            catch_response=True,
        ) as response:
            if response.status_code != 200:
                response.failure(f"unexpected status {response.status_code}: {response.text[:200]}")
                return
            body = response.json()
            if "fraud_probability" not in body or "processing_time_ms" not in body:
                response.failure("missing behavioral response fields")

    @task(6)
    def score_transaction(self):
        with self.client.post(
            "/api/v1/score-transaction",
            json=transaction_payload(self.user_id),
            headers=self.headers,
            name="/api/v1/score-transaction",
            catch_response=True,
        ) as response:
            if response.status_code != 200:
                response.failure(f"unexpected status {response.status_code}: {response.text[:200]}")
                return
            body = response.json()
            if "action" not in body or "processing_time_ms" not in body:
                response.failure("missing transaction response fields")
