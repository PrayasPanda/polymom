"""FastAPI application factory."""

from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.v1.router import api_router
from app.core.config import Settings, get_settings
from app.core.exceptions import error_body, register_exception_handlers
from app.core.logging import configure_logging, get_logger
from app.core.middleware import RequestContextMiddleware, SecurityHeadersMiddleware
from app.core.rate_limit import RateLimiter
from app.core.tracing import setup_tracing
from app.db.migrate import upgrade_to_head
from app.db.session import create_engine, create_sessionmaker
from app.ui import routes as ui

# Allowance for multipart boundaries and form fields on top of the file itself.
MULTIPART_OVERHEAD_BYTES = 64 * 1024


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build and configure the FastAPI application."""
    settings = settings or get_settings()
    configure_logging(settings.log_level, json=settings.app_env != "development")
    logger = get_logger(__name__)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        settings.storage_dir.mkdir(parents=True, exist_ok=True)
        engine = create_engine(settings.resolved_database_url)
        if settings.auto_migrate:
            await upgrade_to_head(engine)
        app.state.engine = engine
        app.state.sessionmaker = create_sessionmaker(engine)
        logger.info(
            "startup",
            app=settings.app_name,
            version=settings.app_version,
            env=settings.app_env,
            pipeline_execution=settings.pipeline_execution,
        )
        yield
        await engine.dispose()
        logger.info("shutdown")

    app = FastAPI(title=settings.app_name, version=settings.app_version, lifespan=lifespan)
    app.dependency_overrides[get_settings] = lambda: settings

    # Correlation-id middleware runs outermost so every log and error carries request_id.
    app.add_middleware(RequestContextMiddleware)
    app.add_middleware(SecurityHeadersMiddleware, hsts=settings.app_env == "production")
    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_credentials=True,
            allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
            allow_headers=["Authorization", "Content-Type", "X-API-Key", "Idempotency-Key"],
            expose_headers=["X-Request-ID", "Retry-After"],
        )

    app.state.rate_limiter = RateLimiter(settings)
    register_exception_handlers(app)

    @app.middleware("http")
    async def reject_oversized_bodies(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Fail fast on a declared Content-Length before any body is read."""
        length = request.headers.get("content-length", "")
        if length.isdigit() and int(length) > settings.max_upload_bytes + MULTIPART_OVERHEAD_BYTES:
            return JSONResponse(
                error_body(
                    "file_too_large",
                    f"File exceeds the {settings.max_upload_mb} MB upload limit.",
                    {"max_upload_mb": settings.max_upload_mb},
                ),
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            )
        json_limit = settings.max_json_body_kb * 1024
        is_json = request.headers.get("content-type", "").startswith("application/json")
        if is_json and length.isdigit() and int(length) > json_limit:
            return JSONResponse(
                error_body(
                    "payload_too_large",
                    f"JSON body exceeds the {settings.max_json_body_kb} KB limit.",
                    {"max_json_body_kb": settings.max_json_body_kb},
                ),
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            )
        return await call_next(request)

    app.include_router(api_router, prefix="/api/v1")
    app.include_router(ui.router)
    app.include_router(ui.partials)
    app.mount("/static/fonts", StaticFiles(directory=ui.FONTS_DIR), name="fonts")
    app.mount("/static", StaticFiles(directory=ui.STATIC_DIR), name="static")
    setup_tracing(settings, app)
    return app


app = create_app()
