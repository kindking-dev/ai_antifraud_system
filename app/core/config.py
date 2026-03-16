from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import computed_field


class Settings(BaseSettings):
    """
    Централизованный менеджер конфигурации приложения.
    Автоматически читает переменные из файла .env.
    Использует строгую типизацию Pydantic V2: если переменная окружения
    отсутствует или имеет неверный тип, приложение упадет при старте (Fail-Fast).
    """

    # Настройка чтения из .env файла
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=True
    )

    # Базовые настройки API
    PROJECT_NAME: str = "AI Behavioral Anti-Fraud API"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/v1"

    # Данные для подключения к PostgreSQL
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_HOST: str
    POSTGRES_PORT: int
    POSTGRES_DB: str

    @computed_field
    @property
    def SQLALCHEMY_DATABASE_URI(self) -> str:
        """
        Динамически собирает URL для подключения к БД.
        Используется асинхронный драйвер `asyncpg` для обеспечения SLA < 50ms.
        """
        return f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

    # Данные для подключения к Redis (Feature Store)
    REDIS_HOST: str
    REDIS_PORT: int

    @computed_field
    @property
    def REDIS_URI(self) -> str:
        """Динамически собирает URL для Redis."""
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/0"


# Создаем синглтон настроек.
# Мы будем импортировать этот объект `settings` в любой файл проекта, где нужны конфигурации.
settings = Settings()
