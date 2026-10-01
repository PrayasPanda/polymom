"""Domain exception hierarchy and their FastAPI handlers.

Every error response uses the envelope::

    {"error": {"code": str, "message": str, "remediation": str | null,
               "details": object | null, "request_id": str | null}}

``code`` is stable (clients switch on it), ``remediation`` tells the user what to
do next. ``retryable`` marks transient failures the job queue may retry
(network, rate limits, timeouts, GPU out-of-memory); validation errors and
corrupted media are never retried. Unexpected exceptions become ``internal_error``
without a stack trace; details are only in the logs, correlated by ``request_id``.
"""

from typing import Any, ClassVar

import structlog
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
    remediation: ClassVar[str | None] = (
        "Retry later; if it persists, contact support with the request_id."
    )
    retryable: ClassVar[bool] = False

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
    remediation = "Check the id; resources owned by another API key are reported as not found."


class MeetingNotFoundError(NotFoundError):
    code = "meeting_not_found"


class ValidationError(PolymomError):
    status_code = status.HTTP_422_UNPROCESSABLE_CONTENT
    code = "validation_error"
    remediation = "Fix the request as described in details and send it again."


class AuthenticationError(PolymomError):
    status_code = status.HTTP_401_UNAUTHORIZED
    code = "unauthorized"
    remediation = "Send a valid API key in the X-API-Key header (python -m scripts.create_api_key)."


class RateLimitedError(PolymomError):
    status_code = status.HTTP_429_TOO_MANY_REQUESTS
    code = "rate_limited"
    remediation = "Wait for the Retry-After seconds, then retry."
    retryable = True


class UnsupportedFileTypeError(PolymomError):
    """Extension not allowed, or content does not match the claimed extension."""

    status_code = status.HTTP_415_UNSUPPORTED_MEDIA_TYPE
    code = "unsupported_file_type"
    remediation = "Upload a real audio or video file (wav, mp3, m4a, flac, ogg, aac, mp4, ...)."


class FileTooLargeError(PolymomError):
    status_code = status.HTTP_413_CONTENT_TOO_LARGE
    code = "file_too_large"
    remediation = "Compress or split the recording, or raise MAX_UPLOAD_MB."


class EmptyFileError(ValidationError):
    code = "empty_file"
    remediation = "The file has no content; upload the recording again."


class CorruptedMediaError(ValidationError):
    """File is unreadable or has no audio stream."""

    code = "corrupted_media"
    remediation = "Re-export the recording with a standard codec (e.g. WAV or MP3) and upload it."


class AudioTooLongError(ValidationError):
    code = "audio_too_long"
    remediation = "Split the recording, or raise MAX_AUDIO_DURATION_MINUTES."


class MediaProbeUnavailableError(PolymomError):
    """ffprobe is missing or failed to start. A server problem, not a client one."""

    code = "media_probe_unavailable"
    remediation = "Install ffmpeg/ffprobe on the server (the Docker image includes it)."


class ConflictError(PolymomError):
    status_code = status.HTTP_409_CONFLICT
    code = "conflict"
    remediation = "Check the resource state (GET the meeting) before retrying."


class MeetingStateConflictError(ConflictError):
    """The meeting is in a state that does not allow the requested action."""

    code = "meeting_state_conflict"


class IdempotencyConflictError(ConflictError):
    code = "idempotency_key_reused"
    remediation = "Use a new Idempotency-Key for a different request."


class PipelineError(PolymomError):
    """Raised when a processing stage (ASR, diarization, ...) fails."""

    code = "pipeline_error"


class AudioProcessingError(PipelineError):
    """ffmpeg failed to decode, filter or write the audio."""

    status_code = status.HTTP_422_UNPROCESSABLE_CONTENT
    code = "audio_processing_failed"
    remediation = "Re-export the recording with a standard codec and upload it again."


class FFmpegTimeoutError(AudioProcessingError):
    status_code = status.HTTP_504_GATEWAY_TIMEOUT
    code = "ffmpeg_timeout"
    remediation = "Retry; for very long recordings raise FFMPEG_TIMEOUT_SECONDS."
    retryable = True


class StageTimeoutError(PipelineError):
    status_code = status.HTTP_504_GATEWAY_TIMEOUT
    code = "stage_timeout"
    remediation = "Retry, or raise the stage limit in STAGE_TIMEOUTS."
    retryable = True


class ResourceExhaustedError(PipelineError):
    """GPU out of memory even after the CPU fallback."""

    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    code = "resource_exhausted"
    remediation = "Retry when the GPU is less busy, lower QUEUE_CONCURRENCY_GPU, or use DEVICE=cpu."


class JobCancelledError(PipelineError):
    status_code = status.HTTP_409_CONFLICT
    code = "cancelled"
    remediation = "Start processing again with POST /meetings/{id}/process."


class DiarizationError(PipelineError):
    """Diarization inference failed."""

    code = "diarization_failed"


class DiarizationModelLoadError(DiarizationError):
    """The diarization model could not be loaded (token, terms, download, extras)."""

    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    code = "diarization_model_unavailable"
    remediation = "Set HF_TOKEN, accept the pyannote model terms, or use DIARIZATION_BACKEND=mock."


class ASRModelLoadError(PipelineError):
    """A speech recognition model could not be loaded."""

    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    code = "asr_model_unavailable"
    remediation = (
        "Install the ML extras (uv sync --extra ml --extra indic) or use ASR_BACKEND=mock."
    )


class UnsupportedLanguageError(PipelineError):
    """No ASR backend is configured for the requested language."""

    status_code = status.HTTP_422_UNPROCESSABLE_CONTENT
    code = "unsupported_language"
    remediation = "Use a supported language hint (en, hi, or) or configure ASR_LANGUAGE_BACKENDS."


class TranscriptionError(PipelineError):
    """ASR inference failed."""

    code = "transcription_failed"


class AudioNotAvailableError(ConflictError):
    """Playback audio was requested after the retention policy purged it."""

    code = "audio_not_available"
    remediation = "Raw audio is purged after RETENTION_DAYS; results stay available."


class TranscriptNotAvailableError(ConflictError):
    """The transcript was requested before transcription finished."""

    code = "transcript_not_available"
    remediation = "Run POST /meetings/{id}/process and wait for GET /status to report completed."


class LanguageIdModelLoadError(PipelineError):
    """A spoken language identification model could not be loaded."""

    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    code = "lid_model_unavailable"
    remediation = "Install the indic extra, pick another LID_BACKEND, or use LID_BACKEND=mock."


class LanguageIdError(PipelineError):
    """Spoken language identification failed."""

    code = "language_id_failed"


class LanguageSummaryNotAvailableError(ConflictError):
    """Language information was requested before language identification ran."""

    code = "language_summary_not_available"
    remediation = "Enable LANGUAGE_ROUTING_ENABLED and process the meeting."


class DiarizationNotAvailableError(ConflictError):
    """Speaker turns were requested before diarization finished."""

    code = "diarization_not_available"
    remediation = "Run POST /meetings/{id}/process and wait for GET /status to report completed."


class AnalyticsNotAvailableError(ConflictError):
    """Analytics were requested before the analytics stage ran."""

    code = "analytics_not_available"
    remediation = "Run POST /meetings/{id}/process and wait for GET /status to report completed."


class ChartsUnavailableError(PolymomError):
    """Charts need the optional ``viz`` extra (matplotlib)."""

    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    code = "charts_unavailable"
    remediation = "Install the viz extra: uv sync --extra viz."


class LLMError(PipelineError):
    """The LLM provider failed: HTTP error, rate limit, timeout."""

    status_code = status.HTTP_502_BAD_GATEWAY
    code = "llm_error"
    remediation = "Retry later with POST /meetings/{id}/summary/regenerate."
    retryable = True


class LLMConfigurationError(LLMError):
    """Missing credentials or endpoint; retrying cannot help."""

    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    code = "llm_not_configured"
    remediation = (
        "Set LLM_API_KEY (and the Azure endpoint settings), or use LLM_PROVIDER=ollama/mock."
    )
    retryable = False


class LLMOutputError(LLMError):
    """The model kept returning output that does not match the schema."""

    code = "llm_invalid_output"
    remediation = "Regenerate with a stronger model (POST /summary/regenerate with model=...)."
    retryable = False


class SummaryNotAvailableError(ConflictError):
    """The summary was requested before (or after a failed) summarization."""

    code = "summary_not_available"
    remediation = "Process the meeting, or retry with POST /meetings/{id}/summary/regenerate."


def current_request_id() -> str | None:
    value = structlog.contextvars.get_contextvars().get("request_id")
    return str(value) if value else None


def error_body(
    code: str,
    message: str,
    details: Any = None,
    remediation: str | None = None,
) -> dict[str, Any]:
    return {
        "error": {
            "code": code,
            "message": message,
            "remediation": remediation,
            "details": jsonable_encoder(details),
            "request_id": current_request_id(),
        }
    }


async def _polymom_error_handler(request: Request, exc: Exception) -> JSONResponse:
    if not isinstance(exc, PolymomError):  # pragma: no cover - registered only for PolymomError
        raise exc
    log = logger.error if exc.status_code >= 500 else logger.warning
    log("request_failed", path=request.url.path, code=exc.code, error=exc.message)
    headers = None
    if isinstance(exc, RateLimitedError) and exc.details and "retry_after" in exc.details:
        headers = {"Retry-After": str(exc.details["retry_after"])}
    return JSONResponse(
        error_body(exc.code, exc.message, exc.details, exc.remediation),
        status_code=exc.status_code,
        headers=headers,
    )


async def _request_validation_handler(request: Request, exc: Exception) -> JSONResponse:
    if not isinstance(exc, RequestValidationError):  # pragma: no cover
        raise exc
    errors = [{k: e[k] for k in ("loc", "msg", "type") if k in e} for e in exc.errors()]
    return JSONResponse(
        error_body(
            "validation_error",
            "Request validation failed.",
            {"errors": errors},
            ValidationError.remediation,
        ),
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


async def _unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Never leak stack traces; the traceback is logged with the request_id."""
    logger.exception("request_crashed", path=request.url.path)
    return JSONResponse(
        error_body(
            "internal_error", "An unexpected error occurred.", None, PolymomError.remediation
        ),
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Attach handlers that map all errors to a consistent JSON envelope."""
    app.add_exception_handler(PolymomError, _polymom_error_handler)
    app.add_exception_handler(RequestValidationError, _request_validation_handler)
    app.add_exception_handler(StarletteHTTPException, _http_exception_handler)
    app.add_exception_handler(Exception, _unhandled_exception_handler)
