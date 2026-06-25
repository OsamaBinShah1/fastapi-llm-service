from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    service_api_key: str = "dev-secret-key"
    default_model: str = "gpt-4o-mini"
    max_tokens: int = 2048
    temperature: float = 0.7

    class Config:
        env_file = ".env"

settings = Settings()
