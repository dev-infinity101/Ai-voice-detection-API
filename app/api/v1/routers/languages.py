from fastapi import APIRouter

from app.schemas.api import LanguagesResponse
from app.services.preprocess import SUPPORTED_LANGUAGES


router = APIRouter()


@router.get("/languages", response_model=LanguagesResponse)
async def languages() -> LanguagesResponse:
    return LanguagesResponse(languages=list(SUPPORTED_LANGUAGES))

