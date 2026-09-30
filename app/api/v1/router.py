"""Aggregate router for API v1.

The meetings and search routers require an API key (``X-API-Key``) whenever
``API_KEY_REQUIRED`` is on, and share the ``RATE_LIMIT_DEFAULT`` budget; health,
metrics and the published schema stay public.
"""

from fastapi import APIRouter, Depends

from app.api.auth import resolve_api_key
from app.api.v1.routes import health, meetings, search, status
from app.core.rate_limit import rate_limit

api_router = APIRouter()
api_router.include_router(health.router)
# Order matters: authenticate first, so the limit is counted per API key.
authenticated = [Depends(resolve_api_key), Depends(rate_limit("default"))]
api_router.include_router(meetings.router, dependencies=authenticated)
api_router.include_router(status.router, dependencies=authenticated)
api_router.include_router(search.router, dependencies=authenticated)
