# Используем официальный легкий образ Python
FROM python:3.10-slim

# Установка системных библиотек (libgomp1 критически важен для работы CatBoost на Linux)
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Устанавливаем рабочую директорию внутри контейнера
WORKDIR /app

# Сначала копируем только requirements, чтобы закэшировать установку библиотек
COPY requirements.txt .

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем весь остальной код и артефакты (папки app и ml_artifacts)
COPY . .

# Открываем порт 8000
EXPOSE 8000

# Команда запуска FastAPI
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]