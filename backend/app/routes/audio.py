import os
import uuid
from typing import List

from fastapi import APIRouter, UploadFile, File, HTTPException

from app.config import settings
from app.diarization.custom_impl import CustomDiarizer  
from app.asr.whisper_asr import WhisperASR
from app.models.schemas import Segment, AnalysisResult

router = APIRouter()
diarizer = CustomDiarizer()
asr = WhisperASR()

os.makedirs(settings.audio_dir, exist_ok=True)

@router.post("/upload", response_model=AnalysisResult)
async def upload_audio(file: UploadFile = File(...)):
   

    filename = file.filename or ""
    if not filename.lower().endswith((".wav", ".mp3")):
        raise HTTPException(status_code=400, detail="Unsupported file type.")

    audio_id = str(uuid.uuid4())
    audio_path = os.path.join(settings.audio_dir, f"{audio_id}.wav")

    # Save uploaded audio
    try:
        with open(audio_path, "wb") as f:
            f.write(await file.read())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save audio: {e}")

    # Diarize
    try:
        diar_segments = diarizer.diarize(audio_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Diarization failed: {e}")

    # ASR 
    try:
        diar_segments = asr.add_transcripts(audio_path, diar_segments)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ASR failed: {e}")

    segments: List[Segment] = [Segment(**seg) for seg in diar_segments]

    return AnalysisResult(
        audio_id=audio_id,
        segments=segments,
    )
