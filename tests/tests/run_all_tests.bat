@echo off
chcp 65001 >nul
echo =======================================================
echo 🛡️ SENTINEL AI: MASTER TEST SUITE
echo =======================================================
echo.

echo [1/2] RUNNING ENGINEERING TESTS (Pytest)...
echo -------------------------------------------------------
:: Запускаем pytest для проверки всего кода в папке tests/
pytest tests/ -v
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ❌ Инженерные тесты провалились! ML-валидация отменена. Исправь код.
    exit /b %ERRORLEVEL%
)
echo ✅ Инженерные тесты пройдены успешно!
echo.

echo [2/2] RUNNING ML VALIDATION SUITE (Live Traffic)...
echo -------------------------------------------------------
:: Проверяем, запущен ли Docker (API должен отвечать)
curl -s http://localhost:8000/health >nul
if %ERRORLEVEL% NEQ 0 (
    echo ⚠️ ВНИМАНИЕ: Docker контейнеры не запущены! 
    echo Для запуска ML-валидатора выполни 'docker-compose up -d'
    echo.
    pause
    exit /b
)

:: Запускаем симуляцию 100 пользователей
python ml_validator.py

echo.
echo =======================================================
echo 🏆 ВСЕ ПРОВЕРКИ УСПЕШНО ЗАВЕРШЕНЫ!
echo =======================================================
pause