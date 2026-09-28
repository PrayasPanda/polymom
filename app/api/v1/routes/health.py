"""Liveness endpoint."""

from fastapi import APIRouter

from app.api.deps import SettingsDep
from app.schemas.health import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health(settings: SettingsDep) -> HealthResponse:
    """Report that the service is up, with its name, version and environment."""
    return HealthResponse(
        status="ok",
        app_name=settings.app_name,
        version=settings.app_version,
        env=settings.app_env,
    )
