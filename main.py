import logging

import base64
import functools
import re
import time

import anyio
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.deps import enforce_api_key
from app.api.v1.router import api_v1_router
from app.core.logging import configure_logging
from app.core.rate_limit import RateLimitMiddleware
from app.core.settings import Settings
from app.schemas.api import VoiceDetectionRequest, VoiceDetectionResponse
from app.services.audio_io import AudioDecodeError, decode_audio_bytes
from app.services.heuristic_detector import HeuristicVoiceDetector
from app.services.preprocess import SUPPORTED_LANGUAGES


def create_app() -> FastAPI:
    settings = Settings.load()
    configure_logging(settings.log_level)

    app = FastAPI(
        title="AI Voice Detection API",
        description="Binary classification: AI-generated vs Human speech",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    app.state.settings = settings
    app.state.detector = HeuristicVoiceDetector(sample_rate=16000)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_allow_origins),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(RateLimitMiddleware, requests_per_minute=settings.rate_limit_rpm)

    app.include_router(api_v1_router, prefix="/api/v1")

    def _decode_audio_base64(raw: str) -> bytes:
        b64_data = raw.strip()
        if b64_data.startswith('"') and b64_data.endswith('"') and len(b64_data) >= 2:
            b64_data = b64_data[1:-1].strip()
        if b64_data.startswith("data:") and "base64," in b64_data:
            b64_data = b64_data.split("base64,", 1)[1]
        elif "," in b64_data:
            b64_data = b64_data.split(",", 1)[-1]
        b64_data = re.sub(r"\s+", "", b64_data)
        padding = (-len(b64_data)) % 4
        if padding:
            b64_data = f"{b64_data}{'=' * padding}"
        try:
            return base64.b64decode(b64_data, validate=True)
        except Exception:
            try:
                return base64.urlsafe_b64decode(b64_data)
            except Exception as e_inner:
                raise HTTPException(status_code=400, detail=f"Invalid base64 audio: {e_inner}") from e_inner

    @app.post("/api/voice-detection", response_model=VoiceDetectionResponse, dependencies=[Depends(enforce_api_key)])
    async def voice_detection(payload: VoiceDetectionRequest, request: Request) -> VoiceDetectionResponse:
        started = time.perf_counter()
        audio_bytes = _decode_audio_base64(payload.audioBase64)

        filename = f"upload.{payload.audioFormat}"
        try:
            decode_fn = functools.partial(
                decode_audio_bytes,
                audio_bytes,
                filename=filename,
                target_sample_rate=16000,
            )
            decoded = await anyio.to_thread.run_sync(decode_fn)
        except AudioDecodeError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to decode audio: {e}") from e

        detector = request.app.state.detector
        result = await anyio.to_thread.run_sync(detector.detect, decoded.waveform, decoded.sample_rate, payload.language)
        processing_ms = (time.perf_counter() - started) * 1000.0

        return VoiceDetectionResponse(
            language=payload.language,
            classification=result.classification,
            confidenceScore=result.confidence_score,
            explanation=result.explanation,
        )

    @app.get("/")
    async def root() -> dict:
        return {
            "service": "AI Voice Detection API",
            "version": "1.0.0",
            "supportedLanguages": list(SUPPORTED_LANGUAGES),
            "docs": "/docs",
            "apiBase": "/api/v1",
        }

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        logging.getLogger("app.http").info(
            "http_error",
            extra={"status_code": exc.status_code, "detail": exc.detail, "path": request.url.path},
        )
        return JSONResponse(status_code=exc.status_code, content={"status": "error", "message": exc.detail})

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        return JSONResponse(status_code=422, content={"status": "error", "message": "Invalid request body"})

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        logging.getLogger("app.unhandled").exception("unhandled_error", extra={"path": request.url.path})
        return JSONResponse(status_code=500, content={"status": "error", "message": "Internal server error"})

    return app


app = create_app()


if __name__ == "__main__":
    import os
    import uvicorn

    settings = Settings.load()
    is_railway = bool(os.getenv("RAILWAY_PROJECT_ID") or os.getenv("RAILWAY_ENVIRONMENT"))
    reload = settings.environment == "dev" and not is_railway
    uvicorn.run("main:app", host=settings.host, port=settings.port, reload=reload)
