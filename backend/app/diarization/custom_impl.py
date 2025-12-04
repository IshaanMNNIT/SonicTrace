from typing import List, Dict
import os

from pyannote.audio import Pipeline
from pyannote.core import Segment as PySegment

from app.diarization.base import Diarizer
from app.config import settings

## Yaha we are inheriting the base Diarizer class and implementing the diarize method using pyannote
class CustomDiarizer(Diarizer):
    

    def __init__(self):
        if not settings.hf_token:
            raise RuntimeError(
                "HuggingFace token is required for pyannote. "
                "No HF_TOKEN found."
            )

        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1,
            use_auth_token=settings.hf_token
        )

    def diarize(self, audio_path: str) -> List[Dict]:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio not found: {audio_path}")

        diarization = self.pipeline(audio_path)

        ## yeh return karega Annotation object. We need to convert it to the desired format.

        segments: List[Dict] = []
        
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            # turn is a pyannote.core.Segment (kab start and kab end)

            start = float(turn.start)
            end = float(turn.end)
            speaker_label = str(speaker)  # Like "SPEAKER_00"

            # Convert -> "Speaker 1", "Speaker 2", ...
            speaker_id = self.speaker_label(speaker_label)

            segments.append(
                {
                    "start": start,
                    "end": end,
                    "speaker": speaker_id,
                }
            )

        segments.sort(key=lambda s: s["start"])
        return segments

    def speaker_label(self, label: str) -> str:
        try:
            idx = int(label.split("_")[-1])
            return f"Speaker {idx + 1}"
        except Exception:
            return label
