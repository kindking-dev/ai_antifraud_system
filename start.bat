@echo off
chcp 65001 >nul
echo ====================================================================
echo 🚀 ЗАПУСК ПРОЕКТА SENTINEL AI + NGROK
echo ====================================================================
echo.

:: Укажите здесь ваш постоянный домен от Ngrok (например: my-domain.ngrok-free.app)
:: Если оставить пустым, запустится обычный случайный адрес
set NGROK_DOMAIN=

echo [1/2] Поднимаем серверную часть (Docker контейнеры)...
docker-compose up -d

echo.
echo [2/2] Ссылки для доступа к сервисам:
echo --------------------------------------------------------------------
echo 📱 МОБИЛЬНОЕ ПРИЛОЖЕНИЕ (Kaspi Clone):
echo    - Локально на ПК:   http://localhost:8080/index.html
if "%NGROK_DOMAIN%"=="" (
    echo    - На телефоне:      [Запустится Ngrok, скопируйте ссылку "Forwarding https://..."] + /index.html
) else (
    echo    - На телефоне:      https://%NGROK_DOMAIN%/index.html
)
echo.
echo 🖥️ ИНСПЕКТОР ТЕЛЕМЕТРИИ (3D Radar):
echo    - Локально на ПК:   http://localhost:8080/inspector.html
if "%NGROK_DOMAIN%"=="" (
    echo    - На телефоне/ПК:   [Запустится Ngrok, скопируйте ссылку "Forwarding https://..."] + /inspector.html
) else (
    echo    - На телефоне/ПК:   https://%NGROK_DOMAIN%/inspector.html
)
echo.
echo 📊 АНАЛИТИКА И МОНИТОРИНГ:
echo    - Дашборд (Streamlit): http://localhost:8501
echo    - Графика (Grafana):   http://localhost:3000 (Вход: admin / admin)
echo    - Метрики (Prometheus):http://localhost:9090
echo    - API Документация:    http://localhost:8000/docs
echo --------------------------------------------------------------------
echo.
echo Для остановки Ngrok нажмите Ctrl+C
echo.

taskkill /F /IM ngrok.exe >nul 2>&1
if "%NGROK_DOMAIN%"=="" (
    ngrok http 8080
) else (
    ngrok http --domain=%NGROK_DOMAIN% 8080
)
