import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.v1.router import api_v1_router
from app.core.logging import configure_logging
from app.core.rate_limit import RateLimitMiddleware
from app.core.settings import Settings
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
