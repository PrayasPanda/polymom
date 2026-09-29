"""Meta MMS language ID (``facebook/mms-lid-126``): the default, covers en, hi and Odia.

VoxLingua107 (SpeechBrain) has no Odia label, so MMS-LID is the maintained
option that recognizes all three meeting languages. Its labels are ISO 639-3
(``eng``, ``hin``, ``ory``); they are mapped to ISO 639-1 here.

**License: CC-BY-NC-4.0 (non-commercial).** Use ``LID_BACKEND=speechbrain`` or
``whisper`` (en/hi only) for commercial deployments.
"""

import importlib
import threading
from pathlib import Path
from typing import Any

from starlette.concurrency import run_in_threadpool

from app.core.config import Settings
from app.core.exceptions import LanguageIdError, LanguageIdModelLoadError
from app.core.logging import get_logger
from app.services.audio.slicing import read_samples
from app.services.language.base import LanguageIdentifier

logger = get_logger(__name__)

ISO3_TO_ISO1 = {
    "eng": "en", "hin": "hi", "ory": "or", "ori": "or", "ben": "bn", "asm": "as",
    "urd": "ur", "mar": "mr", "nep": "ne", "pan": "pa", "guj": "gu", "tel": "te",
    "tam": "ta", "kan": "kn", "mal": "ml", "san": "sa",
}  # fmt: skip

_load_lock = threading.Lock()
_models: dict[tuple[str, str], tuple[Any, Any]] = {}
_inference_lock = threading.Lock()


def _device(settings: Settings, torch: Any) -> str:
    cuda = bool(torch.cuda.is_available())
    if settings.device == "cuda" and not cuda:
        raise LanguageIdModelLoadError(
            "DEVICE=cuda but no CUDA device is available. Use DEVICE=auto or DEVICE=cpu.",
            details={"device": settings.device},
        )
    return "cuda" if settings.device in ("auto", "cuda") and cuda else "cpu"


def load_model(settings: Settings) -> tuple[Any, Any, Any, str]:
    """Shared (feature extractor, model, torch, device); thread-safe lazy load."""
    try:
        torch = importlib.import_module("torch")
        transformers = importlib.import_module("transformers")
    except ImportError as exc:
        raise LanguageIdModelLoadError(
            "MMS language ID needs `uv sync --extra ml --extra indic` "
            "(Docker: INSTALL_ML=true INSTALL_INDIC=true), or set LID_BACKEND=mock.",
            details={"reason": "extras_missing"},
        ) from exc
    device = _device(settings, torch)
    key = (settings.lid_model_id, device)
    with _load_lock:
        if key not in _models:
            logger.info("lid_model_loading", model=settings.lid_model_id, device=device)
            try:
                extractor = transformers.AutoFeatureExtractor.from_pretrained(settings.lid_model_id)
                model = transformers.Wav2Vec2ForSequenceClassification.from_pretrained(
                    settings.lid_model_id
                )
            except Exception as exc:
                raise LanguageIdModelLoadError(
                    f"Could not load {settings.lid_model_id}: {exc}. Check network access to "
                    "Hugging Face and LID_MODEL_ID.",
                    details={"reason": "load_failed", "model": settings.lid_model_id},
                ) from exc
            model.to(device)
            model.eval()
            _models[key] = (extractor, model)
        extractor, model = _models[key]
    return extractor, model, torch, device


class MMSLanguageIdentifier(LanguageIdentifier):
    name = "mms"

    @property
    def model_name(self) -> str:
        return self.settings.lid_model_id

    @property
    def native_languages(self) -> frozenset[str]:
        return frozenset({"en", "hi", "or"})

    def _scores(self, audio_path: Path, start: float, end: float) -> dict[str, float]:
        extractor, model, torch, device = load_model(self.settings)
        samples, rate = read_samples(audio_path, start, end)
        try:
            inputs = extractor(samples, sampling_rate=rate, return_tensors="pt").to(device)
            with _inference_lock, torch.no_grad():
                logits = model(**inputs).logits[0]
            probs = torch.softmax(logits, dim=-1).tolist()
        except Exception as exc:
            raise LanguageIdError(
                f"MMS language ID failed: {exc}", details={"reason": "inference_failed"}
            ) from exc
        labels = model.config.id2label
        return {ISO3_TO_ISO1.get(labels[i], labels[i]): float(p) for i, p in enumerate(probs)}

    async def scores(self, audio_path: Path, start: float, end: float) -> dict[str, float]:
        return await run_in_threadpool(self._scores, audio_path, start, end)
