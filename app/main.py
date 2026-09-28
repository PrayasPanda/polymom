"""FastAPI application factory."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.v1.router import api_router
from app.core.config import Settings, get_settings
from app.core.exceptions import register_exception_handlers
from app.core.logging import configure_logging, get_logger


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build and configure the FastAPI application."""
    settings = settings or get_settings()
    configure_logging(settings.log_level, json=settings.app_env != "development")
    logger = get_logger(__name__)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        settings.storage_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "startup", app=settings.app_name, version=settings.app_version, env=settings.app_env
        )
        # TODO: load ML models here in later prompts.
        yield
        logger.info("shutdown")

    app = FastAPI(title=settings.app_name, version=settings.app_version, lifespan=lifespan)
    app.dependency_overrides[get_settings] = lambda: settings
    register_exception_handlers(app)
    app.include_router(api_router, prefix="/api/v1")
    return app


app = create_app()
