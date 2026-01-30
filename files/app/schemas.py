from pydantic import BaseModel
from typing import Optional

class DetectRequest(BaseModel):
    audio_base64: str
    filename: Optional[str] = "sample.wav"
    language_hint: Optional[str] = None

class DetectResponse(BaseModel):
    classification: str  # "AI" or "Human"
    confidence: float
    language: str
    explanation: str