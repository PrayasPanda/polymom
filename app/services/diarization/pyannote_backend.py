"""pyannote.audio diarization (``pyannote/speaker-diarization-3.1``).

torch and pyannote are optional (``uv sync --extra ml``) and imported lazily, so
the base install and the mock backend never need them. The pipeline is loaded
once per (model, device) and shared; inference runs in a worker thread.
"""

import importlib
import threading
import wave
from pathlib import Path
from typing import Any

from starlette.concurrency import run_in_threadpool

from app.core.config import Settings
from app.core.exceptions import (
    DiarizationError,
    DiarizationModelLoadError,
    ResourceExhaustedError,
)
from app.core.logging import get_logger
from app.services.diarization.base import DiarizationBackend
from app.services.diarization.postprocess import RawDiarization, RawSegment

logger = get_logger(__name__)

TERMS_URLS = (
    "https://huggingface.co/pyannote/speaker-diarization-3.1",
    "https://huggingface.co/pyannote/segmentation-3.0",
)

_load_lock = threading.Lock()
_pipelines: dict[tuple[str, str], Any] = {}
# pyannote pipelines are not safe to call concurrently.
_inference_lock = threading.Lock()


def resolve_device(requested: str, torch: Any) -> str:
    """``auto`` picks CUDA when available; an explicit ``cuda`` must be available."""
    cuda_ok = bool(torch.cuda.is_available())
    if requested == "cuda" and not cuda_ok:
        raise DiarizationModelLoadError(
            "DEVICE=cuda but no CUDA device is available. Use DEVICE=auto or DEVICE=cpu.",
            details={"device": requested},
        )
    if requested == "auto":
        return "cuda" if cuda_ok else "cpu"
    return requested


def _import_ml() -> tuple[Any, Any]:
    try:
        torch = importlib.import_module("torch")
        pyannote_audio = importlib.import_module("pyannote.audio")
    except ImportError as exc:
        raise DiarizationModelLoadError(
            "pyannote.audio is not installed. Install the ML extras with "
            "`uv sync --extra ml` (or build the Docker image with INSTALL_ML=true), "
            "or set DIARIZATION_BACKEND=mock.",
            details={"reason": "ml_extras_missing"},
        ) from exc
    return torch, pyannote_audio


def load_pipeline(settings: Settings) -> Any:
    """Return the shared pyannote pipeline, loading it on first use (thread-safe)."""
    token = settings.hf_token.get_secret_value() if settings.hf_token else ""
    if not token:
        raise DiarizationModelLoadError(
            "HF_TOKEN is not set. Create a Hugging Face access token, accept the model terms "
            f"at {' and '.join(TERMS_URLS)}, then set HF_TOKEN.",
            details={"reason": "missing_hf_token"},
        )
    torch, pyannote_audio = _import_ml()
    device = resolve_device(settings.device, torch)
    key = (settings.diarization_model, device)

    with _load_lock:
        if key in _pipelines:
            return _pipelines[key]
        logger.info("diarization_model_loading", model=settings.diarization_model, device=device)
        try:
            pipeline = pyannote_audio.Pipeline.from_pretrained(
                settings.diarization_model, use_auth_token=token
            )
        except Exception as exc:
            raise DiarizationModelLoadError(
                f"Could not download or load {settings.diarization_model}: {exc}. "
                "Check HF_TOKEN, network access and that the model terms are accepted.",
                details={"reason": "load_failed", "model": settings.diarization_model},
            ) from exc
        if pipeline is None:
            # pyannote returns None (instead of raising) for gated/unauthorized models.
            raise DiarizationModelLoadError(
                f"Access to {settings.diarization_model} was denied. Accept the model terms at "
                f"{' and '.join(TERMS_URLS)} with the account that owns HF_TOKEN.",
                details={"reason": "access_denied", "model": settings.diarization_model},
            )
        pipeline.to(torch.device(device))
        _pipelines[key] = pipeline
        logger.info("diarization_model_loaded", model=settings.diarization_model, device=device)
        return pipeline


def load_waveform(path: Path) -> dict[str, Any]:
    """Read a 16-bit PCM WAV into pyannote's in-memory input format.

    Avoids torchaudio's file I/O, whose API changed across torchaudio versions.
    """
    torch = importlib.import_module("torch")
    numpy = importlib.import_module("numpy")
    with wave.open(str(path), "rb") as wav:
        rate, channels = wav.getframerate(), wav.getnchannels()
        data = wav.readframes(wav.getnframes())
    samples = numpy.frombuffer(data, dtype="<i2").astype("float32") / 32768.0
    waveform = torch.from_numpy(samples.reshape(-1, channels).T.copy())
    return {"waveform": waveform, "sample_rate": rate}


def to_raw(annotation: Any, embeddings: Any) -> RawDiarization:
    """Convert a pyannote ``Annotation`` (+ speaker embeddings) to :class:`RawDiarization`."""
    segments = [
        RawSegment(float(turn.start), float(turn.end), str(label))
        for turn, _, label in annotation.itertracks(yield_label=True)
    ]
    vectors: dict[str, list[float]] = {}
    if embeddings is not None:
        # Rows follow the order of annotation.labels().
        for label, row in zip(annotation.labels(), embeddings, strict=False):
            vectors[str(label)] = [float(x) for x in row]
    return RawDiarization(segments=segments, embeddings=vectors)


def _is_cuda_oom(exc: Exception) -> bool:
    name = type(exc).__name__
    return (
        name in ("OutOfMemoryError", "CudaOutOfMemoryError") or "out of memory" in str(exc).lower()
    )


def _free_cuda_cache() -> None:  # pragma: no cover - only runs with a real GPU
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        logger.warning("cuda_cache_free_failed", exc_info=True)


class PyannoteDiarizationBackend(DiarizationBackend):
    @property
    def model_name(self) -> str:  # type: ignore[override]
        return self.settings.diarization_model

    def _infer(
        self,
        audio_path: Path,
        num_speakers: int | None,
        min_speakers: int | None,
        max_speakers: int | None,
    ) -> RawDiarization:
        pipeline = load_pipeline(self.settings)
        audio = load_waveform(audio_path)
        kwargs = {
            k: v
            for k, v in {
                "num_speakers": num_speakers,
                "min_speakers": min_speakers,
                "max_speakers": max_speakers,
            }.items()
            if v is not None
        }
        try:
            with _inference_lock:
                annotation, embeddings = pipeline(audio, return_embeddings=True, **kwargs)
        except Exception as exc:
            if _is_cuda_oom(exc):
                logger.warning("diarization_cuda_oom_fallback")
                _free_cuda_cache()
                cpu_pipeline = load_pipeline(self.settings.model_copy(update={"device": "cpu"}))
                try:
                    with _inference_lock:
                        annotation, embeddings = cpu_pipeline(
                            audio, return_embeddings=True, **kwargs
                        )
                    from app.core.metrics import OOM_FALLBACKS

                    OOM_FALLBACKS.labels("diarize").inc()
                except Exception as inner:
                    raise ResourceExhaustedError(
                        f"Diarization ran out of memory on GPU and CPU: {inner}",
                        details={"reason": "cuda_oom_after_fallback"},
                    ) from inner
            else:
                raise DiarizationError(
                    f"Diarization failed: {exc}", details={"reason": "inference_failed"}
                ) from exc
        return to_raw(annotation, embeddings)

    async def diarize_raw(
        self,
        audio_path: Path,
        num_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
    ) -> RawDiarization:
        return await run_in_threadpool(
            self._infer, audio_path, num_speakers, min_speakers, max_speakers
        )
