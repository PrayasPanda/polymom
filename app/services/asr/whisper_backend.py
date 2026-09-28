"""faster-whisper backend for English and Hindi (and automatic detection).

``faster_whisper`` is part of the optional ``ml`` extra and imported lazily.
The model is loaded once per (size, device, compute type) and shared; inference
runs in a worker thread with word timestamps and Silero VAD enabled.
"""

import importlib
import math
import threading
import time
from pathlib import Path
from typing import Any

from starlette.concurrency import run_in_threadpool

from app.core.config import Settings
from app.core.exceptions import ASRModelLoadError, TranscriptionError
from app.core.logging import get_logger
from app.schemas.asr import ASRResult, TranscriptSegment, Word
from app.services.asr.base import ASRBackend

logger = get_logger(__name__)

WHISPER_LANGUAGES: frozenset[str] = frozenset({"en", "hi"})

_load_lock = threading.Lock()
_models: dict[tuple[str, str, str], Any] = {}
_inference_lock = threading.Lock()


def resolve_device(requested: str, cuda_available: bool) -> str:
    if requested == "cuda" and not cuda_available:
        raise ASRModelLoadError(
            "DEVICE=cuda but no CUDA device is available. Use DEVICE=auto or DEVICE=cpu.",
            details={"device": requested},
        )
    if requested == "auto":
        return "cuda" if cuda_available else "cpu"
    return requested


def resolve_compute_type(requested: str, device: str) -> str:
    """``auto``: float16 on GPU, int8 on CPU (fast and accurate enough for large-v3)."""
    if requested != "auto":
        return requested
    return "float16" if device == "cuda" else "int8"


def _import_faster_whisper() -> Any:
    try:
        return importlib.import_module("faster_whisper")
    except ImportError as exc:
        raise ASRModelLoadError(
            "faster-whisper is not installed. Install the ML extras with `uv sync --extra ml` "
            "(Docker: INSTALL_ML=true), or set ASR_BACKEND=mock.",
            details={"reason": "ml_extras_missing"},
        ) from exc


def _cuda_available() -> bool:
    try:
        ctranslate2 = importlib.import_module("ctranslate2")
        return bool(ctranslate2.get_cuda_device_count() > 0)
    except Exception:
        return False


def load_model(settings: Settings) -> Any:
    """Shared ``WhisperModel`` (thread-safe lazy load)."""
    faster_whisper = _import_faster_whisper()
    device = resolve_device(settings.device, _cuda_available())
    compute_type = resolve_compute_type(settings.whisper_compute_type, device)
    key = (settings.whisper_model_size, device, compute_type)
    with _load_lock:
        if key in _models:
            return _models[key]
        logger.info(
            "asr_model_loading",
            model=settings.whisper_model_size,
            device=device,
            compute_type=compute_type,
        )
        try:
            model = faster_whisper.WhisperModel(
                settings.whisper_model_size, device=device, compute_type=compute_type
            )
        except Exception as exc:
            raise ASRModelLoadError(
                f"Could not load Whisper '{settings.whisper_model_size}' "
                f"on {device}/{compute_type}: {exc}. Check network access to Hugging Face, "
                "WHISPER_MODEL_SIZE and "
                "WHISPER_COMPUTE_TYPE (float16 needs a GPU).",
                details={"reason": "load_failed", "model": settings.whisper_model_size},
            ) from exc
        _models[key] = model
        return model


def to_segments(
    raw_segments: Any, language: str | None, offset: float, backend: str
) -> list[TranscriptSegment]:
    """Convert faster-whisper segments (with words) to :class:`TranscriptSegment`."""
    segments: list[TranscriptSegment] = []
    for i, seg in enumerate(raw_segments):
        words = [
            Word(
                text=w.word,
                start=round(w.start + offset, 3),
                end=round(w.end + offset, 3),
                confidence=round(float(w.probability), 4),
            )
            for w in (seg.words or [])
        ]
        if words:
            avg = sum(w.confidence or 0.0 for w in words) / len(words)
        else:
            avg = math.exp(seg.avg_logprob) if seg.avg_logprob is not None else 0.0
        segments.append(
            TranscriptSegment(
                id=i,
                start=round(seg.start + offset, 3),
                end=round(seg.end + offset, 3),
                text=seg.text,
                language=language,
                words=words,
                avg_confidence=round(avg, 4),
                backend=backend,
                no_speech_prob=round(float(seg.no_speech_prob), 4),
                compression_ratio=round(float(seg.compression_ratio), 3),
            )
        )
    return segments


class WhisperBackend(ASRBackend):
    name = "whisper"

    @property
    def model_name(self) -> str:
        return f"faster-whisper/{self.settings.whisper_model_size}"

    @property
    def supported_languages(self) -> frozenset[str]:
        return WHISPER_LANGUAGES

    @property
    def supports_auto_detect(self) -> bool:
        return True

    def _run(self, audio_path: Path, language: str | None, offset: float) -> ASRResult:
        started = time.perf_counter()
        model = load_model(self.settings)
        try:
            with _inference_lock:
                raw, info = model.transcribe(
                    str(audio_path),
                    language=language,
                    beam_size=self.settings.asr_beam_size,
                    vad_filter=self.settings.asr_vad_filter,
                    word_timestamps=True,
                    # Not conditioning on previous text avoids Whisper's repetition loops.
                    condition_on_previous_text=False,
                )
                raw_list = list(raw)  # the generator does the actual decoding
        except Exception as exc:
            raise TranscriptionError(
                f"Whisper transcription failed: {exc}", details={"reason": "inference_failed"}
            ) from exc
        detected = language or info.language
        return ASRResult(
            segments=to_segments(raw_list, detected, offset, self.name),
            detected_languages=[],
            model_names=[self.model_name],
            processing_time_ms=int((time.perf_counter() - started) * 1000),
            requested_language=language,
        )

    async def transcribe(
        self, audio_path: Path, language: str | None, offset: float = 0.0
    ) -> ASRResult:
        return await run_in_threadpool(self._run, audio_path, language, offset)
