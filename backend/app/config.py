from pydantic import BaseSettings
class Settings(BaseSettings):
    groq_api_key: str
    hf_token: str
    whisper_model_size: str = "medium"
    device: str = "cuda" 
    audio_dir:str = "data/audio"

    class Config:
        env_file = ".env"
settings = Settings()