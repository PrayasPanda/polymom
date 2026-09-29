import sys
import types
import uuid
from pathlib import Path
from typing import Any

import pytest

from app.core.config import Settings
from app.core.exceptions import (
    LanguageIdError,
    LanguageIdModelLoadError,
    TranscriptionError,
)
from app.schemas.asr import ASRResult
from app.schemas.diarization import SpeakerTurn
from app.schemas.language import LanguageRegion
from app.services.asr.mock_backend import MockASRBackend
from app.services.asr.router import ASRRouter, batch_regions
from app.services.audio.slicing import wav_duration, write_wav_slice
from app.services.language import mms_lid, speechbrain_lid
from app.services.language.base import MIN_MODEL_SECONDS
from app.services.language.mms_lid import MMSLanguageIdentifier
from app.services.language.mock_lid import MockLanguageIdentifier
from app.services.language.service import LanguageIdService, build_identifier
from app.services.language.speechbrain_lid import SpeechBrainLanguageIdentifier
from app.services.language.whisper_lid import WhisperLanguageIdentifier
from tests.conftest import make_wav


def region(start: float, end: float, lang: str, conf: float = 0.9) -> LanguageRegion:
    return LanguageRegion(start=start, end=end, language=lang, speaker=None, confidence=conf)


def wav(tmp_path: Path, seconds: float, name: str = "a.wav") -> Path:
    path = tmp_path / name
    path.write_bytes(make_wav(seconds=seconds))
    return path


# --- batching ---------------------------------------------------------------------------


def test_batch_regions_groups_consecutive_same_language() -> None:
    batches = batch_regions(
        [region(0, 2, "hi", 1.0), region(2, 4, "hi", 0.5), region(4, 6, "or"), region(7, 9, "hi")]
    )

    assert [(b.start, b.end, b.language, len(b.regions)) for b in batches] == [
        (0, 4, "hi", 2),
        (4, 6, "or", 1),
        (7, 9, "hi", 1),
    ]
    assert batches[0].confidence == 0.75


def test_batch_regions_respects_max_span() -> None:
    batches = batch_regions([region(0, 10, "en"), region(10, 20, "en"), region(20, 25, "en")], 20)

    assert [(b.start, b.end) for b in batches] == [(0, 20), (20, 25)]


# --- slicing ----------------------------------------------------------------------------


def test_write_wav_slice_is_sample_accurate(tmp_path: Path) -> None:
    source = wav(tmp_path, 3)

    start = write_wav_slice(source, tmp_path / "s.wav", 1.25, 2.5)

    assert start == 1.25
    assert wav_duration(tmp_path / "s.wav") == 1.25
    assert write_wav_slice(source, tmp_path / "t.wav", 2.5, 10) == 2.5  # clamped to the file
    assert wav_duration(tmp_path / "t.wav") == 0.5


# --- routed transcription -------------------------------------------------------------------


class Failing(MockASRBackend):
    async def transcribe(
        self, audio_path: Path, language: str | None, offset: float = 0.0
    ) -> ASRResult:
        raise TranscriptionError("indic model crashed")


def router(indic: MockASRBackend | None = None) -> ASRRouter:
    settings = Settings(_env_file=None)
    return ASRRouter(
        {
            "whisper": MockASRBackend(settings, route="whisper"),
            "indic": indic or MockASRBackend(settings, route="indic"),
        },
        {"en": "whisper", "hi": "whisper", "or": "indic"},
    )


async def test_transcribe_regions_routes_shifts_and_stitches(tmp_path: Path) -> None:
    audio = wav(tmp_path, 12)
    regions = [region(0, 3, "hi"), region(3, 6, "hi", 0.7), region(6, 9, "or"), region(9, 12, "en")]

    result = await router().transcribe_regions(audio, regions)

    assert [(s.start, s.end, s.language, s.backend) for s in result.segments] == [
        (0, 3, "hi", "mock-whisper"),
        (3, 6, "hi", "mock-whisper"),
        (6, 9, "or", "mock-indic"),
        (9, 12, "en", "mock-whisper"),
    ]
    assert result.segments[0].lid_confidence == 0.8  # batch (0-6 s) confidence
    assert not any(s.fallback_used for s in result.segments)
    assert not [
        p for p in tmp_path.iterdir() if p.name.startswith("asr_regions_")
    ]  # temp slices removed


async def test_odia_failure_falls_back_to_whisper(tmp_path: Path) -> None:
    audio = wav(tmp_path, 6)

    result = await router(Failing(Settings(_env_file=None), route="indic")).transcribe_regions(
        audio, [region(0, 3, "en"), region(3, 6, "or")]
    )

    assert [(s.backend, s.fallback_used) for s in result.segments] == [
        ("mock-whisper", False),
        ("mock-whisper", True),
    ]
    assert result.segments[1].start == 3  # still in meeting time


async def test_failure_of_default_backend_is_not_retried(tmp_path: Path) -> None:
    settings = Settings(_env_file=None)
    failing = ASRRouter({"whisper": Failing(settings)}, {"en": "whisper"})

    with pytest.raises(TranscriptionError):
        await failing.transcribe_regions(wav(tmp_path, 3), [region(0, 3, "en")])


# --- LID service -------------------------------------------------------------------------------


def turns(*spans: tuple[float, float, str]) -> list[SpeakerTurn]:
    return [
        SpeakerTurn(
            speaker_label=s, raw_label="X", start=a, end=b, duration=b - a, is_overlap=False
        )
        for a, b, s in spans
    ]


async def test_lid_service_with_mock_identifier(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, lid_backend="mock")
    audio = wav(tmp_path, 18)
    speakers = turns((0, 6, "Person 1"), (6, 12, "Person 2"), (12, 18, "Person 1"))

    result = await LanguageIdService(MockLanguageIdentifier(settings), settings).identify(
        uuid.uuid4(), audio, speakers
    )

    assert [(r.start, r.end, r.language, r.speaker) for r in result.regions] == [
        (0, 6, "en", "Person 1"),
        (6, 12, "hi", "Person 2"),
        (12, 18, "or", "Person 1"),
    ]
    assert result.summary.num_switches == 2
    assert result.summary.lid_model == "mock-lid"


async def test_single_hint_skips_the_model(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, lid_backend="mms")  # would fail to load without extras

    result = await LanguageIdService(build_identifier(settings), settings).identify(
        uuid.uuid4(), wav(tmp_path, 6), turns((0, 6, "Person 1")), hints=["or"]
    )

    assert [(r.language, r.confidence) for r in result.regions] == [("or", 1.0)]


async def test_identify_short_window_is_uncertain_without_model_call(tmp_path: Path) -> None:
    pred = await MockLanguageIdentifier(Settings(_env_file=None)).identify(
        tmp_path / "x.wav", 0, MIN_MODEL_SECONDS / 2
    )

    assert pred.uncertain


async def test_identify_drops_candidates_the_model_cannot_recognize(tmp_path: Path) -> None:
    class HindiEnglishOnly(MockLanguageIdentifier):
        @property
        def native_languages(self) -> frozenset[str]:
            return frozenset({"en", "hi"})

        async def scores(self, audio_path: Path, start: float, end: float) -> dict[str, float]:
            return {"hi": 0.9, "en": 0.1}

    pred = await HindiEnglishOnly(Settings(_env_file=None)).identify(tmp_path / "x.wav", 0, 5)

    assert {s.language for s in pred.top_k} == {"en", "hi"}


def test_build_identifier() -> None:
    kinds = {
        "mms": MMSLanguageIdentifier,
        "speechbrain": SpeechBrainLanguageIdentifier,
        "whisper": WhisperLanguageIdentifier,
        "mock": MockLanguageIdentifier,
    }
    for name, cls in kinds.items():
        assert isinstance(build_identifier(Settings(_env_file=None, lid_backend=name)), cls)


# --- MMS / SpeechBrain / Whisper LID with faked libraries -----------------------------------------


class FakeTensor:
    def __init__(self, values: list[float]) -> None:
        self.values = values

    def tolist(self) -> list[float]:
        return self.values

    def to(self, device: str) -> "FakeTensor":
        return self

    def __getitem__(self, i: int) -> "FakeTensor":
        return self

    def unsqueeze(self, dim: int) -> "FakeTensor":
        return self


class _NoGrad:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *args: object) -> None:
        return None


def fake_torch(cuda: bool = False) -> types.ModuleType:
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: cuda)  # type: ignore[attr-defined]
    torch.softmax = lambda t, dim: t  # type: ignore[attr-defined]
    torch.no_grad = _NoGrad  # type: ignore[attr-defined]
    torch.from_numpy = lambda x: FakeTensor([])  # type: ignore[attr-defined]
    return torch


@pytest.fixture
def fake_mms(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    state: dict[str, Any] = {"loads": 0, "fail": False, "raise": None}

    class Inputs(dict):  # type: ignore[type-arg]
        def to(self, device: str) -> "Inputs":
            return self

    class Model:
        config = types.SimpleNamespace(id2label={0: "eng", 1: "hin", 2: "ory", 3: "ben"})

        def to(self, device: str) -> None:
            state["device"] = device

        def eval(self) -> None:
            return None

        def __call__(self, **kwargs: Any) -> Any:
            if state["fail"]:
                raise RuntimeError("bad input")
            return types.SimpleNamespace(logits=FakeTensor([0.1, 0.2, 0.5, 0.2]))

    def load(model_id: str) -> Any:
        state["loads"] += 1
        if state["raise"]:
            raise state["raise"]
        return Model()

    transformers = types.ModuleType("transformers")
    transformers.AutoFeatureExtractor = types.SimpleNamespace(  # type: ignore[attr-defined]
        from_pretrained=lambda m: lambda samples, sampling_rate, return_tensors: Inputs()
    )
    transformers.Wav2Vec2ForSequenceClassification = types.SimpleNamespace(from_pretrained=load)  # type: ignore[attr-defined]
    torch = fake_torch()
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setattr(mms_lid, "_models", {})
    monkeypatch.setattr(mms_lid, "read_samples", lambda path, start, end: ([0.0], 16000))
    return state


async def test_mms_maps_iso3_labels_and_renormalizes(
    fake_mms: dict[str, Any], tmp_path: Path
) -> None:
    lid = MMSLanguageIdentifier(Settings(_env_file=None))

    pred = await lid.identify(tmp_path / "a.wav", 0, 3)
    await lid.identify(tmp_path / "a.wav", 3, 6)

    assert (pred.language, pred.confidence) == (
        "or",
        0.625,
    )  # 0.5 / (0.1 + 0.2 + 0.5); Bengali dropped
    assert fake_mms["loads"] == 1
    assert fake_mms["device"] == "cpu"
    assert lid.model_name == "facebook/mms-lid-126"


async def test_mms_errors(fake_mms: dict[str, Any], tmp_path: Path) -> None:
    fake_mms["fail"] = True
    with pytest.raises(LanguageIdError, match="bad input"):
        await MMSLanguageIdentifier(Settings(_env_file=None)).identify(tmp_path / "a.wav", 0, 3)

    fake_mms["raise"] = OSError("offline")
    mms_lid._models.clear()
    with pytest.raises(LanguageIdModelLoadError, match="offline"):
        mms_lid.load_model(Settings(_env_file=None))

    with pytest.raises(LanguageIdModelLoadError, match="no CUDA"):
        mms_lid.load_model(Settings(_env_file=None, device="cuda"))


def test_mms_missing_extras(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "transformers", None)

    with pytest.raises(LanguageIdModelLoadError, match="--extra indic"):
        mms_lid.load_model(Settings(_env_file=None))


@pytest.fixture
def fake_speechbrain(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    state: dict[str, Any] = {"raise": None, "fail": False}

    class Classifier:
        hparams = types.SimpleNamespace(
            label_encoder=types.SimpleNamespace(
                decode_ndim=lambda idx: ["en: English", "hi: Hindi", "bn: Bengali"]
            )
        )

        def classify_batch(self, wavs: Any) -> Any:
            if state["fail"]:
                raise RuntimeError("shape mismatch")
            return (FakeTensor([0.3, 0.6, 0.1]),)

    def from_hparams(source: str, savedir: str, run_opts: dict[str, str]) -> Classifier:
        if state["raise"]:
            raise state["raise"]
        state["run_opts"] = run_opts
        return Classifier()

    classifiers = types.ModuleType("speechbrain.inference.classifiers")
    classifiers.EncoderClassifier = types.SimpleNamespace(from_hparams=from_hparams)  # type: ignore[attr-defined]
    torch = fake_torch()
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "speechbrain.inference.classifiers", classifiers)
    monkeypatch.setattr(speechbrain_lid, "_models", {})
    monkeypatch.setattr(speechbrain_lid, "read_samples", lambda path, start, end: ([0.0], 16000))
    return state


async def test_speechbrain_lid_en_hi_only(fake_speechbrain: dict[str, Any], tmp_path: Path) -> None:
    lid = SpeechBrainLanguageIdentifier(Settings(_env_file=None))

    pred = await lid.identify(tmp_path / "a.wav", 0, 3)

    assert pred.language == "hi"
    assert pred.confidence == pytest.approx(0.6667, abs=1e-4)
    assert {s.language for s in pred.top_k} == {"en", "hi"}  # Odia is not a VoxLingua107 label
    assert fake_speechbrain["run_opts"] == {"device": "cpu"}


async def test_speechbrain_errors(
    fake_speechbrain: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_speechbrain["fail"] = True
    with pytest.raises(LanguageIdError, match="shape mismatch"):
        await SpeechBrainLanguageIdentifier(Settings(_env_file=None)).identify(
            tmp_path / "a.wav", 0, 3
        )

    speechbrain_lid._models.clear()
    fake_speechbrain["raise"] = OSError("offline")
    with pytest.raises(LanguageIdModelLoadError, match="offline"):
        speechbrain_lid.load_model(Settings(_env_file=None))

    monkeypatch.setitem(sys.modules, "speechbrain.inference.classifiers", None)
    with pytest.raises(LanguageIdModelLoadError, match="uv sync --extra ml"):
        speechbrain_lid.load_model(Settings(_env_file=None))


async def test_whisper_lid_uses_all_language_probs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from app.services.asr import whisper_backend
    from app.services.language import whisper_lid

    class Model:
        fail = False

        def transcribe(self, samples: Any, **kwargs: Any) -> Any:
            if self.fail:
                raise RuntimeError("cuda oom")
            return iter([]), types.SimpleNamespace(
                all_language_probs=[("hi", 0.5), ("ur", 0.3), ("en", 0.2)]
            )

    model = Model()
    monkeypatch.setattr(whisper_backend, "load_model", lambda settings: model)
    monkeypatch.setattr(whisper_lid, "read_samples", lambda path, start, end: ([0.0], 16000))
    lid = WhisperLanguageIdentifier(Settings(_env_file=None, whisper_model_size="small"))

    pred = await lid.identify(tmp_path / "a.wav", 0, 3)

    assert pred.language == "hi"
    assert pred.confidence == pytest.approx(0.7143, abs=1e-4)
    assert lid.model_name == "faster-whisper/small:lid"

    model.fail = True
    with pytest.raises(LanguageIdError, match="cuda oom"):
        await lid.identify(tmp_path / "a.wav", 0, 3)


# --- real model (opt-in) ------------------------------------------------------------------------


def _has(module: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(module) is not None


@pytest.mark.slow
@pytest.mark.skipif(not _has("speechbrain"), reason="ml extras not installed")
async def test_real_speechbrain_lid_runs(tmp_path: Path) -> None:
    pred = await SpeechBrainLanguageIdentifier(Settings(_env_file=None)).identify(
        wav(tmp_path, 4), 0, 4
    )

    assert pred.language in {"en", "hi"}
    assert 0 <= pred.confidence <= 1
