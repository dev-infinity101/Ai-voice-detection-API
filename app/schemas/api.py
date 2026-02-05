from typing import Literal

from pydantic import BaseModel, Field


SupportedLanguage = Literal["Tamil", "English", "Hindi", "Malayalam", "Telugu"]


class ErrorResponse(BaseModel):
    status: Literal["error"] = "error"
    message: str


class ClassifyResponse(BaseModel):
    status: Literal["success"] = "success"
    classification: Literal["AI_GENERATED", "HUMAN"]
    confidenceScore: float = Field(..., ge=0.0, le=1.0)
    languageDetected: SupportedLanguage
    probabilities: dict[str, float]
    audioDurationSeconds: float = Field(..., ge=0.0)
    processingMs: float = Field(..., ge=0.0)
    explanation: str


class HealthResponse(BaseModel):
    status: Literal["healthy"] = "healthy"
    cpu: dict[str, str | int | float | bool]
    gpu: dict[str, str | int | float | bool]


class LanguagesResponse(BaseModel):
    languages: list[SupportedLanguage]


class VoiceDetectionRequest(BaseModel):
    language: SupportedLanguage
    audioFormat: Literal["mp3", "wav", "flac"]
    audioBase64: str = Field(..., min_length=1)


class VoiceDetectionResponse(BaseModel):
    status: Literal["success"] = "success"
    language: SupportedLanguage
    classification: Literal["AI_GENERATED", "HUMAN"]
    confidenceScore: float = Field(..., ge=0.0, le=1.0)
    explanation: str
