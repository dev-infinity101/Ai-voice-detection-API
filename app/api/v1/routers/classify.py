import logging
import os
import time
from typing import Annotated

import anyio
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile

from app.api.deps import enforce_api_key
from app.schemas.api import ClassifyResponse
from app.services.audio_io import AudioDecodeError, decode_audio_bytes
from app.services.preprocess import SUPPORTED_LANGUAGES


router = APIRouter()
logger = logging.getLogger("app.classify")


def _validate_filename(filename: str) -> None:
    ext = os.path.splitext(filename.lower())[1]
    if ext not in {".wav", ".flac", ".mp3"}:
        raise HTTPException(status_code=400, detail="Only WAV/FLAC/MP3 files are supported")


@router.post("/classify", response_model=ClassifyResponse, dependencies=[Depends(enforce_api_key)])
async def classify(
    request: Request,
    file: Annotated[UploadFile, File(..., description="WAV/FLAC/MP3 file up to 25 MB")],
    language: Annotated[str, Form(..., description="Tamil|English|Hindi|Malayalam|Telugu")],
) -> ClassifyResponse:
    if language not in SUPPORTED_LANGUAGES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported language. Must be one of: {', '.join(SUPPORTED_LANGUAGES)}",
        )

    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")
    _validate_filename(file.filename)

    max_bytes = int(request.app.state.settings.max_upload_mb) * 1024 * 1024
    raw = await file.read()
    if len(raw) > max_bytes:
        raise HTTPException(status_code=413, detail="File too large")

    try:
        decoded = decode_audio_bytes(raw, filename=file.filename, target_sample_rate=16000)
    except AudioDecodeError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    if decoded.duration_seconds < 0.5:
        raise HTTPException(status_code=400, detail="Audio too short; minimum is 0.5 seconds")
    if decoded.duration_seconds > 60.0:
        raise HTTPException(status_code=400, detail="Audio too long; maximum is 60 seconds")

    detector = request.app.state.detector
    started = time.perf_counter()
    result = await anyio.to_thread.run_sync(detector.detect, decoded.waveform, decoded.sample_rate, language)
    processing_ms = (time.perf_counter() - started) * 1000.0

    logger.info(
        "classified_audio",
        extra={
            "language": language,
            "classification": result.classification,
            "confidence": result.confidence_score,
            "processingMs": processing_ms,
        },
    )

    return ClassifyResponse(
        classification=result.classification,
        confidenceScore=result.confidence_score,
        languageDetected=language,  # type: ignore[arg-type]
        probabilities=result.probabilities,
        audioDurationSeconds=decoded.duration_seconds,
        processingMs=processing_ms,
        explanation=result.explanation,
    )

