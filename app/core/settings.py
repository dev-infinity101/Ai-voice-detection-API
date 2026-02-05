import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    environment: str
    host: str
    port: int
    cors_allow_origins: tuple[str, ...]
    api_key: str | None
    max_upload_mb: int
    rate_limit_rpm: int
    model_path: str
    device: str
    enable_debug_routes: bool
    log_level: str

    @staticmethod
    def load() -> "Settings":
        load_dotenv()

        cors_origins = os.getenv("CORS_ALLOW_ORIGINS", "*").strip()
        allow_origins = tuple(o.strip() for o in cors_origins.split(",") if o.strip())

        api_key = os.getenv("API_KEY", "SleeplessDev3")
        if api_key is not None and not api_key.strip():
            api_key = None

        return Settings(
            environment=os.getenv("ENVIRONMENT", "dev").strip(),
            host=os.getenv("HOST", "0.0.0.0").strip(),
            port=int(os.getenv("PORT", "8000")),
            cors_allow_origins=allow_origins if allow_origins else ("*",),
            api_key=api_key,
            max_upload_mb=int(os.getenv("MAX_UPLOAD_MB", "25")),
            rate_limit_rpm=int(os.getenv("RATE_LIMIT_RPM", "60")),
            model_path=os.getenv("MODEL_PATH", "models/ai_voice_detector.pt"),
            device=os.getenv("DEVICE", "auto").strip().lower(),
            enable_debug_routes=os.getenv("ENABLE_DEBUG_ROUTES", "false").strip().lower()
            in {"1", "true", "yes", "on"},
            log_level=os.getenv("LOG_LEVEL", "INFO").strip().upper(),
        )
