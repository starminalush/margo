from fastapi import APIRouter

from api.endpoints import health, search

router = APIRouter()

router.include_router(health.router, prefix="/health", tags=["health"])
router.include_router(search.router, prefix="/search", tags=["search"])
