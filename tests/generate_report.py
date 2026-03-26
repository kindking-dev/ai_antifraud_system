"""
Validation Report Generator for AI Anti-Fraud System.
Simulates 1000 transactions and generates statistical charts for the thesis.
"""

import httpx
import time
import asyncio
import statistics
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
import uuid
from datetime import datetime, UTC

# Configuration
NUM_REQUESTS = 1000
API_URL = "http://127.0.0.1:8000/v1/score-transaction"

# To store results
latencies = []
actions_count = {"ALLOW": 0, "CHALLENGE": 0, "BLOCK": 0}


async def run_simulation():
    print(f"🚀 Starting simulation of {NUM_REQUESTS} transactions...")

    async with httpx.AsyncClient() as client:
        for i in range(NUM_REQUESTS):
            # 15% трафика - это боты
            is_bot = bool(i % 7 == 0)

            # Разные ID для легитимных юзеров, но одинаковые для ботов (чтобы вызвать Velocity Spike)
            user_id = "BOT-999" if is_bot else f"USR-{uuid.uuid4().hex[:6]}"

            payload = {
                "transaction_id": f"SIM-{i}",
                "user_id": user_id,
                "amount_kzt": float(5000 * (i % 10 + 1)),
                "source": "WEB",
                "session_trust_score": 0.1 if is_bot else 0.95,
                "network": {
                    "ip_address": "1.1.1.1",
                    "is_vpn_or_proxy": is_bot,
                    # Заменили "xyz123" на валидный 32-значный хеш
                    "ja3_fingerprint": "771a4865486602329230abc123456789",
                    "user_agent": "SimBot" if is_bot else "SimUser",
                },
                "biometrics": {
                    "gyroscope_x_y_z": [0.1, 0.1, 0.1] if is_bot else [0.5, 0.6, 0.4],
                    "keystroke_entropy": 0.1 if is_bot else 0.8,
                    "touch_pressure_variance": 0.02 if is_bot else 0.5,
                },
                "timestamp_utc": datetime.now(
                    UTC
                ).isoformat(),  # <-- ВОТ ЭТОГО НЕ ХВАТАЛО
            }

            start = time.perf_counter()
            try:
                resp = await client.post(API_URL, json=payload, timeout=10.0)
                end = time.perf_counter()

                if resp.status_code == 200:
                    data = resp.json()
                    latencies.append((end - start) * 1000)
                    actions_count[data["action"]] += 1
                else:
                    print(f"❌ Error {resp.status_code}: {resp.text}")
            except Exception as e:
                print(f"Request {i} failed: {e}")


def generate_charts():
    print("📊 Generating statistical charts...")
    sns.set_theme(style="whitegrid")

    # 1. График задержек
    plt.figure(figsize=(10, 6))
    sns.histplot(latencies, bins=50, kde=True, color="blue")
    plt.title("API Inference Latency Distribution", fontsize=16)
    plt.xlabel("Latency (ms)", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.axvline(x=50, color="red", linestyle="--", label="SLA Limit (50ms)")
    plt.legend()
    plt.savefig("tests/latency_dist.png")
    plt.close()

    # 2. Распределение решений системы
    plt.figure(figsize=(8, 6))
    plt.bar(
        list(actions_count.keys()),
        list(actions_count.values()),
        color=["#a2fca2", "#fcd3a2", "#fca2a2"],
    )
    plt.title("Decision Distribution (Normal vs Fraud)", fontsize=16)
    plt.ylabel("Number of Transactions", fontsize=12)
    plt.savefig("tests/action_dist.png")
    plt.close()


def create_pdf_report():
    print("📝 Compiling PDF Validation Report...")
    pdf = FPDF()
    pdf.add_page()

    # Заголовок
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, txt="AI Anti-Fraud System Validation Report", ln=True, align="C")
    pdf.ln(10)

    # Статистика
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, txt=f"Total Transactions Simulated: {len(latencies)}", ln=True)
    pdf.cell(
        0, 10, txt=f"Average Latency: {statistics.mean(latencies):.2f} ms", ln=True
    )
    pdf.cell(
        0,
        10,
        txt=f"P99 Latency: {statistics.quantiles(latencies, n=100)[98]:.2f} ms",
        ln=True,
    )
    pdf.cell(
        0,
        10,
        txt=f"Decisions: ALLOW={actions_count['ALLOW']}, CHALLENGE={actions_count['CHALLENGE']}, BLOCK={actions_count['BLOCK']}",
        ln=True,
    )

    pdf.ln(10)

    # Вставка графиков
    pdf.cell(0, 10, txt="1. System Performance (Latency)", ln=True)
    pdf.image("tests/latency_dist.png", x=10, w=150)

    pdf.add_page()
    pdf.cell(0, 10, txt="2. ML Detection Accuracy (Action Distribution)", ln=True)
    pdf.image("tests/action_dist.png", x=10, w=150)

    pdf.output("tests/Validation_Report.pdf")
    print("✅ Report successfully saved as 'tests/Validation_Report.pdf'")


if __name__ == "__main__":
    # Запускаем асинхронный симулятор
    asyncio.run(run_simulation())

    # Собираем отчеты, только если были успешные запросы
    if latencies:
        generate_charts()
        create_pdf_report()
    else:
        print("❌ No successful transactions recorded. Check server logs.")
