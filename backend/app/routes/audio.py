import os
import uuid
import traceback
from typing import List

from fastapi import APIRouter, UploadFile, File, HTTPException

from app.config import settings
from app.diarization.silero_impl import SileroDiarizer
from app.asr.whisper_asr import WhisperASR
from app.models.schemas import Segment, AnalysisResult

router = APIRouter()

diarizer = SileroDiarizer()
asr = WhisperASR()

os.makedirs(settings.audio_dir, exist_ok=True)

ALLOWED_EXTS = {".wav", ".mp3", ".m4a", ".flac"}


@router.post("/upload", response_model=AnalysisResult)
async def upload_audio(file: UploadFile = File(...)):
    """
    Upload an audio file -> run diarization (Silero + MFCC pipeline) -> align ASR text.
    Notes:
     - We preserve the original file extension when saving.
     - If ASR fails, we still return diarization segments with empty text.
    """
    filename = file.filename or ""
    ext = os.path.splitext(filename)[1].lower()

    if ext not in ALLOWED_EXTS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

    audio_id = str(uuid.uuid4())
    # keep original extension so downstream decoders can detect format
    audio_path = os.path.join(settings.audio_dir, f"{audio_id}{ext}")

    # Save uploaded audio (raw bytes)
    try:
        with open(audio_path, "wb") as f:
            f.write(await file.read())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save audio: {e}")

    # 1) Diarize (Silero + features + clustering)
    try:
        diar_segments = diarizer.diarize(audio_path)
    except Exception as e:
        # return the error to client with some detail
        tb = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"Diarization failed: {e}\n{tb}")

    # initialize text fields in case ASR is skipped or fails
    for seg in diar_segments:
        seg["text"] = ""

    # 2) ASR (Whisper): try but don't fail the whole request if it errors
    try:
        diar_segments = asr.add_transcripts(audio_path, diar_segments)
    except Exception as e:
        # Log the ASR failure, keep segments with empty texts
        print(f"[WARN] ASR failed: {e}")
        print(traceback.format_exc())
        # ensure text keys exist
        for seg in diar_segments:
            seg.setdefault("text", "")

    # Build response
    segments: List[Segment] = [Segment(**seg) for seg in diar_segments]

    return AnalysisResult(
        audio_id=audio_id,
        segments=segments,
    )