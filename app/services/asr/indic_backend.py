"""Odia ASR with AI4Bharat IndicConformer (``ai4bharat/indic-conformer-600m-multilingual``).

The model ships as ONNX + TorchScript with ``trust_remote_code`` and is called
as ``model(waveform, "or", "ctc" | "rnnt") -> str``. It returns **text only**, so:

* segment timestamps come from energy-based speech regions (frame accurate, 30 ms),
* word timestamps are approximated inside each segment in proportion to word
  length (see :func:`~app.services.asr.postprocess.approximate_words`).

Needs the ``ml`` + ``indic`` extras and an ``HF_TOKEN`` whose account accepted the
model's terms (the repository is gated).
"""

import importlib
import threading
import time
import wave
from pathlib import Path
from typing import Any

from starlette.concurrency import run_in_threadpool

from app.core.config import Settings
from app.core.exceptions import ASRModelLoadError, TranscriptionError
from app.core.logging import get_logger
from app.schemas.asr import ASRResult, TranscriptSegment
from app.services.asr.base import ASRBackend
from app.services.asr.postprocess import approximate_words
from app.services.asr.segmentation import FRAME_SECONDS, speech_regions

logger = get_logger(__name__)

INDIC_LANGUAGES: frozenset[str] = frozenset({"or"})
MODEL_URL = "https://huggingface.co/{model_id}"

_load_lock = threading.Lock()
_models: dict[str, Any] = {}
_inference_lock = threading.Lock()


def load_model(settings: Settings) -> Any:
    """Shared IndicConformer model (thread-safe lazy load)."""
    token = settings.hf_token.get_secret_value() if settings.hf_token else None
    url = MODEL_URL.format(model_id=settings.odia_model_id)
    if not token:
        raise ASRModelLoadError(
            f"HF_TOKEN is not set. {settings.odia_model_id} is gated: accept its terms at {url} "
            "and set HF_TOKEN, or route Odia elsewhere via ASR_LANGUAGE_BACKENDS.",
            details={"reason": "missing_hf_token", "model": settings.odia_model_id},
        )
    try:
        transformers = importlib.import_module("transformers")
        importlib.import_module("onnxruntime")
    except ImportError as exc:
        raise ASRModelLoadError(
            "Odia ASR needs the indic extras: `uv sync --extra ml --extra indic` "
            "(Docker: INSTALL_ML=true INSTALL_INDIC=true).",
            details={"reason": "indic_extras_missing"},
        ) from exc

    with _load_lock:
        if settings.odia_model_id in _models:
            return _models[settings.odia_model_id]
        logger.info("asr_model_loading", model=settings.odia_model_id)
        try:
            model = transformers.AutoModel.from_pretrained(
                settings.odia_model_id, trust_remote_code=True, token=token
            )
        except Exception as exc:
            raise ASRModelLoadError(
                f"Could not load {settings.odia_model_id}: {exc}. Check HF_TOKEN, that the "
                f"model terms are accepted at {url}, and network access.",
                details={"reason": "load_failed", "model": settings.odia_model_id},
            ) from exc
        _models[settings.odia_model_id] = model
        return model


def load_audio(path: Path) -> tuple[Any, int, list[float]]:
    """Samples (float32 numpy), sample rate, and per-frame energy in dBFS."""
    numpy = importlib.import_module("numpy")
    with wave.open(str(path), "rb") as wav:
        rate = wav.getframerate()
        data = wav.readframes(wav.getnframes())
    samples = numpy.frombuffer(data, dtype="<i2").astype("float32") / 32768.0
    hop = int(rate * FRAME_SECONDS)
    frames = len(samples) // hop
    if frames == 0:
        return samples, rate, []
    rms = numpy.sqrt(numpy.mean(samples[: frames * hop].reshape(frames, hop) ** 2, axis=1))
    energies = (20 * numpy.log10(numpy.maximum(rms, 1e-10))).tolist()
    return samples, rate, [float(e) for e in energies]


class IndicConformerBackend(ASRBackend):
    name = "indic"

    @property
    def model_name(self) -> str:
        return f"{self.settings.odia_model_id}:{self.settings.odia_decoding}"

    @property
    def supported_languages(self) -> frozenset[str]:
        return INDIC_LANGUAGES

    def _run(self, audio_path: Path, language: str | None, offset: float) -> ASRResult:
        started = time.perf_counter()
        lang = language or "or"
        model = load_model(self.settings)
        torch = importlib.import_module("torch")
        samples, rate, energies = load_audio(audio_path)

        segments: list[TranscriptSegment] = []
        try:
            for start, end in speech_regions(energies):
                piece = samples[int(start * rate) : int(end * rate)]
                waveform = torch.from_numpy(piece.copy()).unsqueeze(0)
                with _inference_lock:
                    text = str(model(waveform, lang, self.settings.odia_decoding))
                seg_start, seg_end = start + offset, end + offset
                segments.append(
                    TranscriptSegment(
                        id=len(segments),
                        start=round(seg_start, 3),
                        end=round(seg_end, 3),
                        text=text,
                        language=lang,
                        words=approximate_words(text, seg_start, seg_end),
                        avg_confidence=None,  # the model exposes no scores
                        backend=self.name,
                    )
                )
        except Exception as exc:
            raise TranscriptionError(
                f"IndicConformer transcription failed: {exc}",
                details={"reason": "inference_failed"},
            ) from exc
        return ASRResult(
            segments=segments,
            detected_languages=[],
            model_names=[self.model_name],
            processing_time_ms=int((time.perf_counter() - started) * 1000),
            requested_language=language,
        )

    async def transcribe(
        self, audio_path: Path, language: str | None, offset: float = 0.0
    ) -> ASRResult:
        return await run_in_threadpool(self._run, audio_path, language, offset)
