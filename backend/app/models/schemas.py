from typing import List
from pydantic import BaseModel

class Segment(BaseModel):
    start: float
    end: float
    speaker: str
    text: str

class AnalysisResult(BaseModel):
    audio_id : str
    segments : List[Segment]