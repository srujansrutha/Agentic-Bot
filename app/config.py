from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    model_name: str
    db_path: str = "chatbot.db"
    mongo_uri: str
    mongo_db_name: str = "chatbot"
    redis_url: str = "redis://localhost:6380"
    embedding_model: str = "qwen3-embedding:8b"


settings = Settings()
