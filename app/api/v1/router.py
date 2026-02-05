from fastapi import APIRouter

from app.api.v1.routers import classify, debug, health, languages


api_v1_router = APIRouter()

api_v1_router.include_router(health.router, tags=["health"])
api_v1_router.include_router(languages.router, tags=["languages"])
api_v1_router.include_router(classify.router, tags=["classification"])
api_v1_router.include_router(debug.router, tags=["debug"])

