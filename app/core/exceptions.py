"""Domain exception hierarchy and their FastAPI handlers."""

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from app.core.logging import get_logger

logger = get_logger(__name__)


class PolymomError(Exception):
    """Base class for all application errors."""

    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR
    code: str = "internal_error"

    def __init__(self, message: str = "An unexpected error occurred.") -> None:
        super().__init__(message)
        self.message = message


class NotFoundError(PolymomError):
    status_code = status.HTTP_404_NOT_FOUND
    code = "not_found"


class ValidationError(PolymomError):
    status_code = status.HTTP_422_UNPROCESSABLE_CONTENT
    code = "validation_error"


class UnsupportedMediaError(PolymomError):
    status_code = status.HTTP_415_UNSUPPORTED_MEDIA_TYPE
    code = "unsupported_media"


class PayloadTooLargeError(PolymomError):
    status_code = status.HTTP_413_CONTENT_TOO_LARGE
    code = "payload_too_large"


class PipelineError(PolymomError):
    """Raised when a processing stage (ASR, diarization, ...) fails."""

    code = "pipeline_error"


async def _polymom_error_handler(request: Request, exc: Exception) -> JSONResponse:
    if not isinstance(exc, PolymomError):  # pragma: no cover - registered only for PolymomError
        raise exc
    logger.warning("request_failed", path=request.url.path, code=exc.code, error=exc.message)
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"code": exc.code, "message": exc.message}},
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Attach handlers that map domain errors to a consistent JSON envelope."""
    app.add_exception_handler(PolymomError, _polymom_error_handler)
