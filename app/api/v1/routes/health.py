"""Liveness and readiness. ``/health`` stays for backwards compatibility."""

import importlib.util
from typing import Any

import sqlalchemy as sa
from fastapi import APIRouter
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.responses import JSONResponse, Response

from app.api.deps import ArtifactStoreDep, SettingsDep, UowDep
from app.core.config import Settings
from app.core.logging import get_logger
from app.core.metrics import QUEUE_DEPTH
from app.schemas.health import HealthResponse
from app.workers.queue import QUEUE_NAMES, build_queue
from app.workers.redis_client import build_redis

logger = get_logger(__name__)

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health(settings: SettingsDep) -> HealthResponse:
    """Kept for existing clients; new deployments prefer /health/live and /health/ready."""
    return HealthResponse(
        status="ok",
        app_name=settings.app_name,
        version=settings.app_version,
        env=settings.app_env,
    )


@router.get("/health/live", summary="Liveness: the process is up")
async def live() -> dict[str, str]:
    return {"status": "ok"}


@router.get(
    "/health/ready",
    summary="Readiness: DB, Redis (if configured), and artifact store are usable",
)
async def ready(
    uow: UowDep,
    store: ArtifactStoreDep,
    settings: SettingsDep,
) -> Response:
    checks: dict[str, dict[str, Any]] = {}
    ok = True

    try:
        await uow.session.execute(sa.text("SELECT 1"))
        checks["database"] = {"ok": True}
    except Exception as exc:
        ok = False
        checks["database"] = {"ok": False, "error": str(exc)[:200]}

    try:
        exists = await store.exists("meetings/__ready_check__")
        checks["artifact_store"] = {"ok": True, "sample_exists": exists}
    except Exception as exc:
        ok = False
        checks["artifact_store"] = {"ok": False, "error": str(exc)[:200]}

    if settings.redis_url:
        try:
            redis = build_redis(settings)
            try:
                await redis.ping()
                checks["redis"] = {"ok": True}
            finally:
                await redis.close()
        except Exception as exc:
            ok = False
            checks["redis"] = {"ok": False, "error": str(exc)[:200]}
    else:
        checks["redis"] = {"ok": True, "configured": False}

    # Informational: workers load the models, the API does not, so a missing model
    # does not make the API unready (workers report failures on their jobs).
    checks["models"] = model_availability(settings)
    body = {"status": "ok" if ok else "unavailable", "checks": checks}
    return JSONResponse(body, status_code=200 if ok else 503)


def model_availability(settings: Settings) -> dict[str, Any]:
    """Whether the configured ML backends are installed (and credentialed) here."""

    def installed(module: str) -> bool:
        try:
            return importlib.util.find_spec(module) is not None
        except (ImportError, ValueError):
            return False

    backends = {
        "diarization": settings.diarization_backend == "mock"
        or (installed("pyannote.audio") and settings.hf_token is not None),
        "asr": settings.asr_backend == "mock" or installed("faster_whisper"),
        "language_id": settings.lid_backend == "mock" or installed("transformers"),
        "llm": settings.llm_provider in ("mock", "ollama") or settings.llm_api_key is not None,
    }
    return {"ok": all(backends.values()), **backends}


@router.get("/metrics", summary="Prometheus metrics", include_in_schema=False)
async def metrics(settings: SettingsDep) -> Response:
    """API metrics, plus queue depth per queue when the job queue is in use."""
    if settings.pipeline_execution == "queue" and settings.redis_url:
        queue = build_queue(settings)
        try:
            for name in QUEUE_NAMES:
                QUEUE_DEPTH.labels(name).set(await queue.queue_depth(name))
        except Exception:  # never fail a scrape because Redis is down
            logger.warning("queue_depth_unavailable", exc_info=True)
        finally:
            await queue.close()
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
