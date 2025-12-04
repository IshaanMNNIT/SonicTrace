import os

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.config import settings

router = APIRouter()


@router.get("/audio/{audio_id}")
def get_audio(audio_id: str):
    
    audio_path = os.path.join(settings.audio_dir, f"{audio_id}.wav")
    if not os.path.exists(audio_path):
        raise HTTPException(status_code=404, detail="Audio not found.")
    return FileResponse(audio_path, media_type="audio/wav")