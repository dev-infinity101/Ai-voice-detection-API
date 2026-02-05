import os
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile

from app.api.deps import enforce_api_key
from app.services.audio_io import AudioDecodeError, decode_audio_bytes
from app.services.preprocess import SUPPORTED_LANGUAGES, preprocess_waveform


router = APIRouter(prefix="/_debug")


def _ensure_enabled(request: Request) -> None:
    if not request.app.state.settings.enable_debug_routes:
        raise HTTPException(status_code=404, detail="Not found")


def _validate_filename(filename: str) -> None:
    ext = os.path.splitext(filename.lower())[1]
    if ext not in {".wav", ".flac", ".mp3"}:
        raise HTTPException(status_code=400, detail="Only WAV/FLAC/MP3 files are supported")


@router.post("/upload", dependencies=[Depends(enforce_api_key)])
async def upload_audio(
    request: Request,
    file: Annotated[UploadFile, File(...)],
) -> dict:
    _ensure_enabled(request)
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")
    _validate_filename(file.filename)

    max_bytes = int(request.app.state.settings.max_upload_mb) * 1024 * 1024
    raw = await file.read()
    if len(raw) > max_bytes:
        raise HTTPException(status_code=413, detail="File too large")
    return {"filename": file.filename, "bytes": len(raw)}


@router.post("/features", dependencies=[Depends(enforce_api_key)])
async def extract_features(
    request: Request,
    file: Annotated[UploadFile, File(...)],
    language: Annotated[str, Form(...)],
) -> dict:
    _ensure_enabled(request)
    if language not in SUPPORTED_LANGUAGES:
        raise HTTPException(status_code=400, detail="Unsupported language")
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")
    _validate_filename(file.filename)

    raw = await file.read()
    try:
        decoded = decode_audio_bytes(raw, filename=file.filename, target_sample_rate=16000)
    except AudioDecodeError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    detector = request.app.state.detector
    y = preprocess_waveform(decoded.waveform, decoded.sample_rate, language=language)
    features = detector.extract_audio_features(y, decoded.sample_rate)
    return {"features": features}


@router.post("/infer", dependencies=[Depends(enforce_api_key)])
async def infer(
    request: Request,
    file: Annotated[UploadFile, File(...)],
    language: Annotated[str, Form(...)],
) -> dict:
    _ensure_enabled(request)
    if language not in SUPPORTED_LANGUAGES:
        raise HTTPException(status_code=400, detail="Unsupported language")
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")
    _validate_filename(file.filename)

    raw = await file.read()
    try:
        decoded = decode_audio_bytes(raw, filename=file.filename, target_sample_rate=16000)
    except AudioDecodeError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    detector = request.app.state.detector
    result = detector.detect(decoded.waveform, decoded.sample_rate, language=language)
    return {
        "classification": result.classification,
        "confidenceScore": result.confidence_score,
        "probabilities": result.probabilities,
        "explanation": result.explanation,
    }
