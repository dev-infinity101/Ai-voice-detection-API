import os
import platform

from fastapi import APIRouter

from app.schemas.api import HealthResponse


router = APIRouter()


def build_health_response() -> HealthResponse:
    try:
        import torch  # type: ignore
    except Exception:
        torch = None

    cpu = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "threads": int(torch.get_num_threads()) if torch else 0,
        "pid": os.getpid(),
    }

    if torch and torch.cuda.is_available():
        gpu = {
            "available": True,
            "deviceCount": torch.cuda.device_count(),
            "deviceName": torch.cuda.get_device_name(0),
        }
    else:
        gpu = {"available": False}

    return HealthResponse(status="healthy", cpu=cpu, gpu=gpu)


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return build_health_response()
