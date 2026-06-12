"""
SENTINEL AI: Master Test Orchestrator.
Runs engineering tests (pytest) followed by the ML validation suite.
"""

import subprocess
import sys
import urllib.request
import time

def run_tests():
    print("=" * 60)
    print(" SENTINEL AI: MASTER TEST SUITE")
    print("=" * 60)

    # Шаг 1: Инженерные тесты (Pytest)
    print("\n[1/2] RUNNING ENGINEERING TESTS (Pytest)...")
    print("-" * 60)
    
    # Запускаем pytest и ждем завершения
    pytest_result = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-v"])
    
    if pytest_result.returncode != 0:
        print("\n[ERROR] Инженерные тесты провалились! ML-валидация отменена. Исправь ошибки в коде.")
        sys.exit(pytest_result.returncode)
        
    print("[OK] Инженерные тесты пройдены успешно!\n")

    # Шаг 2: Проверка доступности Docker
    print("[2/2] RUNNING ML VALIDATION SUITE (Live Traffic)...")
    print("-" * 60)
    try:
        urllib.request.urlopen("http://localhost:8000/health", timeout=3)
    except Exception:
        print("\n[WARNING] Docker контейнеры не отвечают!")
        print("Убедись, что выполнил команду: docker-compose up -d")
        sys.exit(1)

    # Шаг 3: Запуск ML-валидатора
    subprocess.run([sys.executable, "ml_validator.py"])

    print("\n" + "=" * 60)
    print(" ВСЕ ПРОВЕРКИ УСПЕШНО ЗАВЕРШЕНЫ!")
    print("=" * 60)

if __name__ == "__main__":
    try:
        run_tests()
    except KeyboardInterrupt:
        print("\nТестирование прервано пользователем.")
        sys.exit(0)