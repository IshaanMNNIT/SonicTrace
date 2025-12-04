from typing import List, Dict

import whisper

from app.config import settings


class WhisperASR:

    def __init__(self):
        self.device = settings.device
        self.model = whisper.load_model(
            settings.whisper_model_size,
            device=self.device
        )

    @staticmethod
    def _overlap_duration(
        s1: float, e1: float,
        s2: float, e2: float
    ) -> float:
        start = max(s1, s2)
        end = min(e1, e2)
        return max(0.0, end - start)

    def add_transcripts(self, audio_path: str, segments: List[Dict]) -> List[Dict]:
        """
        Given an audio file and a list of diarized segments of the form:

            {
              "start": float,
              "end": float,
              "speaker": "Speaker 1",
              ...
            }
        Returns the same list of segments, with "text" fields filled in.
        """

        ## Single full-audio transcription
        result = self.model.transcribe(
            audio_path,
            language=None,
            fp16=(self.device == "cuda"),
        )

        asr_segments = result.get("segments", []) or []

        for seg in segments:
            seg["text"] = ""

        for a_seg in asr_segments:
            a_start = float(a_seg["start"])
            a_end = float(a_seg["end"])
            text = a_seg.get("text", "").strip()
            if not text:
                continue

            best_overlap_duration = 0.0
            best_match_seg: Dict | None = None

            #  max overlap ?? - find the diarization segment with the most overlap
            for d_seg in segments:
                d_start = d_seg["start"]
                d_end = d_seg["end"]

                overlap = self._overlap_duration(d_start, d_end, a_start, a_end)
                if overlap > best_overlap_duration:
                    best_overlap_duration = overlap
                    best_match_seg = d_seg

            # Only assign if there is a meaningful overlap aur agar pehle se text hua to usme add kar do
            if best_overlap_duration > 0.1 and best_match_seg is not None:
                current_text = best_match_seg.get("text", "")
                best_match_seg["text"] = (
                    f"{current_text} {text}".strip() if current_text else text
                )

        return segments
