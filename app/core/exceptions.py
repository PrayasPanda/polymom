"""Domain exception hierarchy and their FastAPI handlers.

Every error response uses the envelope
``{"error": {"code": str, "message": str, "details": object | null}}``.
"""

from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.logging import get_logger

logger = get_logger(__name__)


class PolymomError(Exception):
    """Base class for all application errors."""

    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR
    code: str = "internal_error"

    def __init__(
        self,
        message: str = "An unexpected error occurred.",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.details = details


class NotFoundError(PolymomError):
    status_code = status.HTTP_404_NOT_FOUND
    code = "not_found"


class MeetingNotFoundError(NotFoundError):
    code = "meeting_not_found"


class ValidationError(PolymomError):
    status_code = status.HTTP_422_UNPROCESSABLE_CONTENT
    code = "validation_error"


class UnsupportedFileTypeError(PolymomError):
    """Extension not allowed, or content does not match the claimed extension."""

    status_code = status.HTTP_415_UNSUPPORTED_MEDIA_TYPE
    code = "unsupported_file_type"


class FileTooLargeError(PolymomError):
    status_code = status.HTTP_413_CONTENT_TOO_LARGE
    code = "file_too_large"


class EmptyFileError(ValidationError):
    code = "empty_file"


class CorruptedMediaError(ValidationError):
    """File is unreadable or has no audio stream."""

    code = "corrupted_media"


class MediaProbeUnavailableError(PolymomError):
    """ffprobe is missing or failed to start. A server problem, not a client one."""

    code = "media_probe_unavailable"


class ConflictError(PolymomError):
    status_code = status.HTTP_409_CONFLICT
    code = "conflict"


class MeetingStateConflictError(ConflictError):
    """The meeting is in a state that does not allow the requested action."""

    code = "meeting_state_conflict"


class PipelineError(PolymomError):
    """Raised when a processing stage (ASR, diarization, ...) fails."""

    code = "pipeline_error"


class AudioProcessingError(PipelineError):
    """ffmpeg failed to decode, filter or write the audio."""

    status_code = status.HTTP_422_UNPROCESSABLE_CONTENT
    code = "audio_processing_failed"


class FFmpegTimeoutError(AudioProcessingError):
    status_code = status.HTTP_504_GATEWAY_TIMEOUT
    code = "ffmpeg_timeout"


class DiarizationError(PipelineError):
    """Diarization inference failed."""

    code = "diarization_failed"


class DiarizationModelLoadError(DiarizationError):
    """The diarization model could not be loaded (token, terms, download, extras)."""

    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    code = "diarization_model_unavailable"


class ASRModelLoadError(PipelineError):
    """A speech recognition model could not be loaded."""

    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    code = "asr_model_unavailable"


class UnsupportedLanguageError(PipelineError):
    """No ASR backend is configured for the requested language."""

    status_code = status.HTTP_422_UNPROCESSABLE_CONTENT
    code = "unsupported_language"


class TranscriptionError(PipelineError):
    """ASR inference failed."""

    code = "transcription_failed"


class TranscriptNotAvailableError(ConflictError):
    """The transcript was requested before transcription finished."""

    code = "transcript_not_available"


class LanguageIdModelLoadError(PipelineError):
    """A spoken language identification model could not be loaded."""

    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    code = "lid_model_unavailable"


class LanguageIdError(PipelineError):
    """Spoken language identification failed."""

    code = "language_id_failed"


class LanguageSummaryNotAvailableError(ConflictError):
    """Language information was requested before language identification ran."""

    code = "language_summary_not_available"


class DiarizationNotAvailableError(ConflictError):
    """Speaker turns were requested before diarization finished."""

    code = "diarization_not_available"


class AnalyticsNotAvailableError(ConflictError):
    """Analytics were requested before the analytics stage ran."""

    code = "analytics_not_available"


class ChartsUnavailableError(PolymomError):
    """Charts need the optional ``viz`` extra (matplotlib)."""

    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    code = "charts_unavailable"


def error_body(code: str, message: str, details: Any = None) -> dict[str, Any]:
    return {"error": {"code": code, "message": message, "details": jsonable_encoder(details)}}


async def _polymom_error_handler(request: Request, exc: Exception) -> JSONResponse:
    if not isinstance(exc, PolymomError):  # pragma: no cover - registered only for PolymomError
        raise exc
    log = logger.error if exc.status_code >= 500 else logger.warning
    log("request_failed", path=request.url.path, code=exc.code, error=exc.message)
    return JSONResponse(error_body(exc.code, exc.message, exc.details), status_code=exc.status_code)


async def _request_validation_handler(request: Request, exc: Exception) -> JSONResponse:
    if not isinstance(exc, RequestValidationError):  # pragma: no cover
        raise exc
    errors = [{k: e[k] for k in ("loc", "msg", "type") if k in e} for e in exc.errors()]
    return JSONResponse(
        error_body("validation_error", "Request validation failed.", {"errors": errors}),
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
    )


async def _http_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    if not isinstance(exc, StarletteHTTPException):  # pragma: no cover
        raise exc
    return JSONResponse(
        error_body(f"http_{exc.status_code}", str(exc.detail)),
        status_code=exc.status_code,
        headers=exc.headers,
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Attach handlers that map all errors to a consistent JSON envelope."""
    app.add_exception_handler(PolymomError, _polymom_error_handler)
    app.add_exception_handler(RequestValidationError, _request_validation_handler)
    app.add_exception_handler(StarletteHTTPException, _http_exception_handler)
