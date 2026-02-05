from fastapi import Header, HTTPException


def require_api_key(configured_api_key: str | None, x_api_key: str | None = Header(None)) -> None:
    if configured_api_key is None:
        return

    if not x_api_key:
        raise HTTPException(
            status_code=401,
            detail="Missing API key. Include 'x-api-key' in request headers.",
        )

    if x_api_key != configured_api_key:
        raise HTTPException(status_code=403, detail="Invalid API key")

