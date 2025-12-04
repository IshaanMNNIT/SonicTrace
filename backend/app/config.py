from pydantic_settings import BaseSettings, SettingsConfigDict
class Settings(BaseSettings):
    hf_token: str | None = None
    groq_api_key: str | None = None
    whisper_model_size: str = "medium"   
    device: str = "cuda"
    audio_dir: str = "data/audio"
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )
settings = Settings()
