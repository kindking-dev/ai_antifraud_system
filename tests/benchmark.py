import httpx
import time
import statistics
import asyncio

async def run_benchmark(n_requests=100):
    latencies = []
    results = []
    
    async with httpx.AsyncClient() as client:
        for i in range(n_requests):
            # Симулируем микс из нормальных и подозрительных транзакций
            is_bot = i % 5 == 0 
            payload = {
                "transaction_id": f"TEST-{i}",
                "user_id": "TEST-USER",
                "amount_kzt": 5000 * (i + 1),
                "source": "WEB",
                "session_trust_score": 0.1 if is_bot else 0.9,
                "network": {
                    "ip_address": "1.1.1.1",
                    "is_vpn_or_proxy": is_bot,
                    "ja3_fingerprint": "771a4865486602329230abc123456789",
                    "user_agent": "PyTest"
                },
                "biometrics": {"gyroscope_x_y_z": [0.1, 0.1, 0.1] if is_bot else [0.5, 0.5, 0.5], "keystroke_entropy": 0.1 if is_bot else 0.8, "touch_pressure_variance": 0.01 if is_bot else 0.1}
            }

            start = time.perf_counter()
            resp = await client.post("http://127.0.0.1:8000/v1/score-transaction", json=payload)
            end = time.perf_counter()
            
            if resp.status_code == 200:
                latencies.append((end - start) * 1000)
                results.append(resp.json())

    print(f"📈 BENCHMARK RESULTS ({n_requests} requests):")
    print(f"✅ Avg Latency: {statistics.mean(latencies):.2f} ms")
    print(f"🚀 P99 Latency: {statistics.quantiles(latencies, n=100)[98]:.2f} ms")
    print(f"🛡️ High Risk Detected: {len([r for r in results if r['fraud_probability'] > 0.8])} cases")

if __name__ == "__main__":
    asyncio.run(run_benchmark())