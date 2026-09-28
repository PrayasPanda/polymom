"""SpeechBrain VoxLingua107 ECAPA language ID (``speechbrain/lang-id-voxlingua107-ecapa``).

Apache-2.0 and fast, but **VoxLingua107 has no Odia label** (verified against the
model's ``label_encoder.txt``): it can only choose between English and Hindi
here. Odia speech would be labelled Hindi/Bengali, so use it for en/hi meetings
or commercial deployments where MMS-LID's non-commercial license is a problem.
"""

import importlib
import os
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

_load_lock = threading.Lock()
_models: dict[tuple[str, str], Any] = {}
_inference_lock = threading.Lock()


def load_model(settings: Settings) -> tuple[Any, Any]:
    """Shared (classifier, torch); thread-safe lazy load."""
    try:
        torch = importlib.import_module("torch")
        classifiers = importlib.import_module("speechbrain.inference.classifiers")
    except ImportError as exc:
        raise LanguageIdModelLoadError(
            "SpeechBrain is not installed: `uv sync --extra ml` (Docker: INSTALL_ML=true), "
            "or set LID_BACKEND=mock.",
            details={"reason": "extras_missing"},
        ) from exc
    cuda = bool(torch.cuda.is_available())
    device = "cuda" if settings.device in ("auto", "cuda") and cuda else "cpu"
    key = (settings.speechbrain_lid_model_id, device)
    with _load_lock:
        if key not in _models:
            savedir = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
            try:
                _models[key] = classifiers.EncoderClassifier.from_hparams(
                    source=settings.speechbrain_lid_model_id,
                    savedir=str(savedir / "speechbrain" / "lang-id-voxlingua107-ecapa"),
                    run_opts={"device": device},
                )
            except Exception as exc:
                raise LanguageIdModelLoadError(
                    f"Could not load {settings.speechbrain_lid_model_id}: {exc}.",
                    details={"reason": "load_failed"},
                ) from exc
        return _models[key], torch


class SpeechBrainLanguageIdentifier(LanguageIdentifier):
    name = "speechbrain"

    @property
    def model_name(self) -> str:
        return self.settings.speechbrain_lid_model_id

    @property
    def native_languages(self) -> frozenset[str]:
        return frozenset({"en", "hi"})

    def _scores(self, audio_path: Path, start: float, end: float) -> dict[str, float]:
        classifier, torch = load_model(self.settings)
        samples, _ = read_samples(audio_path, start, end)
        try:
            with _inference_lock, torch.no_grad():
                log_probs = classifier.classify_batch(torch.from_numpy(samples).unsqueeze(0))[0][0]
            probs = torch.softmax(log_probs, dim=-1).tolist()
        except Exception as exc:
            raise LanguageIdError(
                f"SpeechBrain language ID failed: {exc}", details={"reason": "inference_failed"}
            ) from exc
        labels = classifier.hparams.label_encoder.decode_ndim(list(range(len(probs))))
        # Labels look like "hi: Hindi".
        return {
            str(label).split(":", 1)[0].strip(): float(p)
            for label, p in zip(labels, probs, strict=True)
        }

    async def scores(self, audio_path: Path, start: float, end: float) -> dict[str, float]:
        return await run_in_threadpool(self._scores, audio_path, start, end)
