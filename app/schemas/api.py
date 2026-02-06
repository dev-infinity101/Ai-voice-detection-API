from typing import Annotated, Literal

from pydantic import AliasChoices, BaseModel, BeforeValidator, Field


SupportedLanguage = Literal["Tamil", "English", "Hindi", "Malayalam", "Telugu"]


def strip_whitespace(v: str) -> str:
    if isinstance(v, str):
        return v.strip()
    return v


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
    language: Annotated[SupportedLanguage, BeforeValidator(strip_whitespace)]
    audioFormat: Annotated[Literal["mp3", "wav", "flac"], BeforeValidator(strip_whitespace)]
    audioBase64: Annotated[
        str,
        BeforeValidator(strip_whitespace),
        Field(
            ...,
            min_length=1,
            validation_alias=AliasChoices("audioBase64", "audioBase64Format"),
            serialization_alias="audioBase64",
        ),
    ]


class VoiceDetectionResponse(BaseModel):
    status: Literal["success"] = "success"
    language: SupportedLanguage
    classification: Literal["AI_GENERATED", "HUMAN"]
    confidenceScore: float = Field(..., ge=0.0, le=1.0)
    explanation: str
