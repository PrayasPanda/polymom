"""Aggregate router for API v1.

The meetings and search routers require an API key (``X-API-Key``) whenever
``API_KEY_REQUIRED`` is on; the health, metrics and schema routes stay public.
"""

from fastapi import APIRouter, Depends

from app.api.auth import resolve_api_key
from app.api.v1.routes import health, meetings, search, status

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(meetings.router, dependencies=[Depends(resolve_api_key)])
api_router.include_router(status.router, dependencies=[Depends(resolve_api_key)])
api_router.include_router(search.router, dependencies=[Depends(resolve_api_key)])
