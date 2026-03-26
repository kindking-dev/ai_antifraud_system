"""
Local Development Orchestrator.
Starts both FastAPI backend and Streamlit dashboard concurrently.
"""

import subprocess
import sys
import time


def main():
    print("🚀 Starting AI Anti-Fraud System Development Environment...")
    print("-" * 50)

    # 1. Запуск FastAPI (Бэкенд)
    print("📦 [1/2] Starting FastAPI Backend on port 8000...")
    backend = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--reload", "--port", "8000"]
    )

    # Даем бэкенду пару секунд на загрузку ML-модели в RAM
    time.sleep(3)

    # 2. Запуск Streamlit (Дашборд)
    print("📊 [2/2] Starting Streamlit Dashboard...")
    frontend = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "app/services/dashboard.py"]
    )

    print("-" * 50)
    print("✅ System is LIVE! Press Ctrl+C to stop all services.")

    try:
        # Держим процесс активным
        backend.wait()
        frontend.wait()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down services safely...")
        backend.terminate()
        frontend.terminate()
        print("✅ System offline. Goodbye!")


if __name__ == "__main__":
    main()
