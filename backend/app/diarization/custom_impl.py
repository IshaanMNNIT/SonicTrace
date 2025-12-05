from typing import List, Dict
import os

from torch.serialization import add_safe_globals
from torch.torch_version import TorchVersion
import pyannote.audio.core.task as task_mod
from pyannote.audio import Pipeline

from app.diarization.base import Diarizer
from app.config import settings


# ---- PyTorch 2.6+ / 2.9+ safe unpickling fix for pyannote ----
# 1) Allow TorchVersion (used in some checkpoints)
safe_globals = [TorchVersion]

# 2) Allow ALL classes defined in pyannote.audio.core.task
for name, obj in vars(task_mod).items():
    if isinstance(obj, type):
        safe_globals.append(obj)

add_safe_globals(safe_globals)
# ---------------------------------------------------------------


class CustomDiarizer(Diarizer):
    """
    Wrapper around pyannote's speaker diarization pipeline (v3.x).

    Uses: pyannote/speaker-diarization-3.1
    """

    def __init__(self):
        if not settings.hf_token:
            raise RuntimeError(
                "HF_TOKEN is required for pyannote models. Set it in your .env"
            )

        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=settings.hf_token,
        )

    def diarize(self, audio_path: str) -> List[Dict]:
        """
        Run diarization on an audio file path and return:
        [
          { "start": float, "end": float, "speaker": "Speaker 1" },
          ...
        ]
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio not found: {audio_path}")

        output = self.pipeline(audio_path)

        annotation = output.speaker_diarization

        segments: List[Dict] = []
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            segments.append(
                {
                    "start": float(turn.start),
                    "end": float(turn.end),
                    "speaker": self._normalize(str(speaker)),
                }
            )

        return sorted(segments, key=lambda s: s["start"])

    def _normalize(self, speaker_label: str) -> str:
        """
        Converts SPEAKER_00, SPEAKER_01, etc. to Speaker 1, Speaker 2, etc.
        """
        try:
            n = int(speaker_label.split("_")[-1])
            return f"Speaker {n + 1}"
        except Exception:
            return speaker_label
