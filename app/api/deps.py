from fastapi import Header, HTTPException, Request

from app.core.settings import Settings


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def enforce_api_key(request: Request, x_api_key: str | None = Header(None)) -> None:
    settings: Settings = request.app.state.settings
    if settings.api_key is None:
        return

    if not x_api_key:
        raise HTTPException(
            status_code=401,
            detail="Missing API key. Include 'x-api-key' in request headers.",
        )

    if x_api_key != settings.api_key:
        raise HTTPException(status_code=403, detail="Invalid API key")

