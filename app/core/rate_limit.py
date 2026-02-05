import asyncio
import time
from collections import deque
from dataclasses import dataclass

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response


@dataclass
class _Bucket:
    timestamps: deque[float]


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, requests_per_minute: int) -> None:
        super().__init__(app)
        self._rpm = max(int(requests_per_minute), 1)
        self._lock = asyncio.Lock()
        self._buckets: dict[str, _Bucket] = {}

    async def dispatch(self, request: Request, call_next) -> Response:
        client = request.client
        ip = client.host if client else "unknown"

        now = time.monotonic()
        cutoff = now - 60.0

        async with self._lock:
            bucket = self._buckets.get(ip)
            if bucket is None:
                bucket = _Bucket(timestamps=deque())
                self._buckets[ip] = bucket

            while bucket.timestamps and bucket.timestamps[0] < cutoff:
                bucket.timestamps.popleft()

            if len(bucket.timestamps) >= self._rpm:
                return JSONResponse(
                    status_code=429,
                    content={
                        "status": "error",
                        "message": "Rate limit exceeded",
                        "limit": self._rpm,
                        "windowSeconds": 60,
                    },
                )

            bucket.timestamps.append(now)

        return await call_next(request)

