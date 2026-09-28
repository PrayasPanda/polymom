import os
import sys
import types
import unicodedata
import urllib.request
import uuid
import wave
from pathlib import Path
from typing import Any, ClassVar

import pytest

from app.core.config import Settings
from app.core.exceptions import ASRModelLoadError, TranscriptionError
from app.schemas.asr import ASRResult, TranscriptSegment
from app.services.asr import indic_backend, whisper_backend
from app.services.asr.base import ASRBackend
from app.services.asr.indic_backend import IndicConformerBackend
from app.services.asr.mock_backend import MockASRBackend
from app.services.asr.router import ASRRouter
from app.services.asr.service import TranscriptionService
from app.services.asr.whisper_backend import (
    WhisperBackend,
    resolve_compute_type,
    resolve_device,
)
from tests.conftest import make_wav


def write(path: Path, data: bytes) -> Path:
    path.write_bytes(data)
    return path


def silent(path: Path, seconds: float = 2.0) -> Path:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * int(16000 * seconds))
    return path


# --- mock -------------------------------------------------------------------------


async def test_mock_forced_language_uses_that_script(tmp_path: Path) -> None:
    result = await MockASRBackend(Settings(_env_file=None)).transcribe(
        write(tmp_path / "a.wav", make_wav(seconds=7)), "or", offset=100
    )

    assert [s.language for s in result.segments] == ["or", "or", "or"]
    assert result.segments[0].start == 100
    assert result.segments[-1].end == 107
    assert not unicodedata.is_normalized("NFC", result.segments[0].text)  # raw NFD on purpose


async def test_mock_rotates_languages_without_hint(tmp_path: Path) -> None:
    result = await MockASRBackend(Settings(_env_file=None)).transcribe(
        write(tmp_path / "a.wav", make_wav(seconds=9)), None
    )

    assert [s.language for s in result.segments] == ["en", "hi", "or"]


async def test_mock_silence_gives_no_segments(tmp_path: Path) -> None:
    result = await MockASRBackend(Settings(_env_file=None)).transcribe(
        silent(tmp_path / "s.wav"), "en"
    )

    assert result.segments == []


# --- service ------------------------------------------------------------------------


async def test_service_normalizes_filters_and_reports_languages(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, storage_dir=tmp_path, asr_backend="mock")

    class Noisy(MockASRBackend):
        async def transcribe(
            self, audio_path: Path, language: str | None, offset: float = 0.0
        ) -> ASRResult:
            result = await super().transcribe(audio_path, language, offset)
            ghost = TranscriptSegment(
                id=99, start=8.0, end=9.0, text="Thanks for watching!", language="en",
                words=[], avg_confidence=0.4, backend="mock", no_speech_prob=0.95,
            )  # fmt: skip
            return result.model_copy(update={"segments": [*result.segments, ghost]})

    backend = Noisy(settings)
    service = TranscriptionService(ASRRouter({"whisper": backend}, {"hi": "whisper"}), settings)

    result = await service.transcribe(
        uuid.uuid4(), write(tmp_path / "a.wav", make_wav(seconds=6)), ["hi"]
    )

    assert [s.id for s in result.segments] == [0, 1]
    assert all(unicodedata.is_normalized("NFC", s.text) for s in result.segments)
    assert all(s.language == "hi" for s in result.segments)
    assert "Thanks for watching!" not in [s.text for s in result.segments]
    assert [(d.language, d.duration_seconds) for d in result.detected_languages] == [("hi", 6.0)]
    assert result.requested_language == "hi"
    assert result.model_names == ["mock-asr"]


async def test_service_chunks_long_audio_without_duplicates(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,
        storage_dir=tmp_path,
        asr_backend="mock",
        chunk_length_seconds=12,
        chunk_overlap_seconds=3,
    )
    service = TranscriptionService(
        ASRRouter({"whisper": MockASRBackend(settings)}, {"en": "whisper"}), settings
    )

    result = await service.transcribe(
        uuid.uuid4(), write(tmp_path / "long.wav", make_wav(seconds=30)), ["en"]
    )

    starts = [s.start for s in result.segments]
    assert starts == sorted(starts)
    for a, b in zip(result.segments, result.segments[1:], strict=False):
        assert b.start >= a.end - 0.001  # no overlapping duplicates survive
    assert result.segments[0].start == 0
    assert result.segments[-1].end == 30
    assert not any(p.name.endswith("_asr_chunks") for p in tmp_path.iterdir())


# --- whisper (faster_whisper faked) ----------------------------------------------


class FakeWord:
    def __init__(self, word: str, start: float, end: float, probability: float) -> None:
        self.word, self.start, self.end, self.probability = word, start, end, probability


class FakeSegment:
    def __init__(self, text: str, start: float, end: float, words: list[FakeWord] | None) -> None:
        self.text, self.start, self.end, self.words = text, start, end, words
        self.avg_logprob, self.no_speech_prob, self.compression_ratio = -0.2, 0.01, 1.3


class FakeWhisperModel:
    instances: ClassVar[list["FakeWhisperModel"]] = []

    def __init__(self, size: str, device: str, compute_type: str) -> None:
        self.size, self.device, self.compute_type = size, device, compute_type
        self.calls: list[dict[str, Any]] = []
        self.fail = False
        FakeWhisperModel.instances.append(self)

    def transcribe(self, path: str, **kwargs: Any) -> tuple[Any, Any]:
        if self.fail:
            raise RuntimeError("decoder exploded")
        self.calls.append(kwargs)
        words = [FakeWord(" Hello", 0.0, 0.6, 0.9), FakeWord(" there.", 0.6, 1.5, 0.7)]
        segments = [
            FakeSegment(" Hello there.", 0.0, 1.5, words),
            FakeSegment(" Bye.", 2.0, 2.5, None),
        ]
        return iter(segments), types.SimpleNamespace(language="en", language_probability=0.98)


@pytest.fixture
def fake_whisper(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    state: dict[str, Any] = {"cuda": 0, "raise": None}
    FakeWhisperModel.instances = []

    def factory(size: str, device: str, compute_type: str) -> FakeWhisperModel:
        if state["raise"]:
            raise state["raise"]
        return FakeWhisperModel(size, device, compute_type)

    module = types.ModuleType("faster_whisper")
    module.WhisperModel = factory  # type: ignore[attr-defined]
    ct2 = types.ModuleType("ctranslate2")
    ct2.get_cuda_device_count = lambda: state["cuda"]  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "faster_whisper", module)
    monkeypatch.setitem(sys.modules, "ctranslate2", ct2)
    monkeypatch.setattr(whisper_backend, "_models", {})
    return state


async def test_whisper_backend_with_fakes(fake_whisper: dict[str, Any], tmp_path: Path) -> None:
    backend = WhisperBackend(Settings(_env_file=None, whisper_model_size="small", asr_beam_size=3))

    result = await backend.transcribe(tmp_path / "a.wav", None, offset=10)
    await backend.transcribe(tmp_path / "a.wav", "hi")

    model = FakeWhisperModel.instances[0]
    assert len(FakeWhisperModel.instances) == 1  # loaded once
    assert (model.size, model.device, model.compute_type) == ("small", "cpu", "int8")
    assert model.calls[0] == {
        "language": None, "beam_size": 3, "vad_filter": True,
        "word_timestamps": True, "condition_on_previous_text": False,
    }  # fmt: skip
    assert model.calls[1]["language"] == "hi"
    first, second = result.segments
    assert (first.start, first.end, first.language) == (10.0, 11.5, "en")  # detected language
    assert [(w.text, w.start, w.confidence) for w in first.words] == [
        (" Hello", 10.0, 0.9),
        (" there.", 10.6, 0.7),
    ]
    assert first.avg_confidence == 0.8
    assert second.avg_confidence == pytest.approx(0.8187, abs=1e-4)  # exp(avg_logprob), no words
    assert (first.no_speech_prob, first.compression_ratio) == (0.01, 1.3)
    assert backend.model_name == "faster-whisper/small"
    assert backend.supported_languages == {"en", "hi"}


def test_whisper_uses_cuda_float16_when_available(fake_whisper: dict[str, Any]) -> None:
    fake_whisper["cuda"] = 1

    whisper_backend.load_model(Settings(_env_file=None))

    assert FakeWhisperModel.instances[0].compute_type == "float16"
    assert FakeWhisperModel.instances[0].device == "cuda"


def test_whisper_load_failure_is_actionable(fake_whisper: dict[str, Any]) -> None:
    fake_whisper["raise"] = ValueError("float16 unsupported")

    with pytest.raises(ASRModelLoadError, match="float16 unsupported") as exc:
        whisper_backend.load_model(Settings(_env_file=None))
    assert exc.value.status_code == 503


def test_whisper_missing_extras(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "faster_whisper", None)

    with pytest.raises(ASRModelLoadError, match="uv sync --extra ml"):
        whisper_backend.load_model(Settings(_env_file=None))


async def test_whisper_inference_failure(fake_whisper: dict[str, Any], tmp_path: Path) -> None:
    backend = WhisperBackend(Settings(_env_file=None))
    whisper_backend.load_model(backend.settings).fail = True

    with pytest.raises(TranscriptionError, match="decoder exploded"):
        await backend.transcribe(tmp_path / "a.wav", "en")


@pytest.mark.parametrize(
    ("requested", "cuda", "expected"),
    [("auto", True, "cuda"), ("auto", False, "cpu"), ("cpu", True, "cpu")],
)
def test_resolve_device(requested: str, cuda: bool, expected: str) -> None:
    assert resolve_device(requested, cuda) == expected


def test_resolve_device_cuda_missing() -> None:
    with pytest.raises(ASRModelLoadError, match="no CUDA device"):
        resolve_device("cuda", False)


def test_resolve_compute_type() -> None:
    assert resolve_compute_type("auto", "cuda") == "float16"
    assert resolve_compute_type("auto", "cpu") == "int8"
    assert resolve_compute_type("float32", "cuda") == "float32"


def test_cuda_probe_without_ctranslate2(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "ctranslate2", None)

    assert whisper_backend._cuda_available() is False


# --- IndicConformer (transformers/onnxruntime/torch faked) ---------------------------


class FakeTensor:
    def __init__(self, data: Any) -> None:
        self.data = data

    def unsqueeze(self, dim: int) -> "FakeTensor":
        return self


class FakeIndicModel:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self.fail = False

    def __call__(self, waveform: FakeTensor, lang: str, decoding: str) -> str:
        if self.fail:
            raise RuntimeError("onnx session crashed")
        self.calls.append((lang, decoding))
        return ["ନମସ୍କାର ସମସ୍ତଙ୍କୁ", "ଆଜି ବଜେଟ୍"][len(self.calls) - 1]


class FakeSamples(list):  # type: ignore[type-arg]
    def copy(self) -> "FakeSamples":
        return self


@pytest.fixture
def fake_indic(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    state: dict[str, Any] = {"model": FakeIndicModel(), "raise": None, "loads": 0}

    def from_pretrained(model_id: str, trust_remote_code: bool, token: str) -> FakeIndicModel:
        state["loads"] += 1
        state["kwargs"] = {"trust_remote_code": trust_remote_code, "token": token}
        if state["raise"]:
            raise state["raise"]
        return state["model"]  # type: ignore[no-any-return]

    transformers = types.ModuleType("transformers")
    transformers.AutoModel = types.SimpleNamespace(from_pretrained=from_pretrained)  # type: ignore[attr-defined]
    torch = types.ModuleType("torch")
    torch.from_numpy = FakeTensor  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "onnxruntime", types.ModuleType("onnxruntime"))
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(indic_backend, "_models", {})
    # 16 kHz: speech 0.3-2.0 s and 3.0-4.5 s (30 ms frames), silence elsewhere.
    energies = [-60.0] * 10 + [-20.0] * 57 + [-60.0] * 33 + [-20.0] * 50 + [-60.0] * 10
    monkeypatch.setattr(
        indic_backend, "load_audio", lambda path: (FakeSamples([0.0] * 80000), 16000, energies)
    )
    return state


def hf(**kwargs: Any) -> Settings:
    return Settings(_env_file=None, hf_token="hf_test", **kwargs)


async def test_indic_backend_segments_from_speech_regions(
    fake_indic: dict[str, Any], tmp_path: Path
) -> None:
    backend = IndicConformerBackend(hf(odia_decoding="ctc"))

    result = await backend.transcribe(tmp_path / "a.wav", "or", offset=60)

    assert [(s.start, s.end, s.text) for s in result.segments] == [
        (60.3, 62.01, "ନମସ୍କାର ସମସ୍ତଙ୍କୁ"),
        (63.0, 64.5, "ଆଜି ବଜେଟ୍"),
    ]
    first = result.segments[0]
    assert (first.language, first.backend, first.avg_confidence) == ("or", "indic", None)
    assert [w.text for w in first.words] == ["ନମସ୍କାର", "ସମସ୍ତଙ୍କୁ"]
    assert (first.words[0].start, first.words[-1].end) == (60.3, 62.01)
    assert fake_indic["model"].calls == [("or", "ctc"), ("or", "ctc")]
    assert fake_indic["kwargs"] == {"trust_remote_code": True, "token": "hf_test"}
    assert backend.model_name == "ai4bharat/indic-conformer-600m-multilingual:ctc"
    assert backend.supported_languages == {"or"}
    assert not backend.supports_auto_detect


def test_indic_missing_token() -> None:
    with pytest.raises(ASRModelLoadError, match="HF_TOKEN is not set") as exc:
        indic_backend.load_model(Settings(_env_file=None))
    assert "huggingface.co/ai4bharat/indic-conformer-600m-multilingual" in exc.value.message


def test_indic_missing_extras(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "transformers", None)

    with pytest.raises(ASRModelLoadError, match="--extra indic"):
        indic_backend.load_model(hf())


def test_indic_load_failure_and_caching(fake_indic: dict[str, Any]) -> None:
    indic_backend.load_model(hf())
    indic_backend.load_model(hf())
    assert fake_indic["loads"] == 1

    indic_backend._models.clear()
    fake_indic["raise"] = OSError("403 gated")
    with pytest.raises(ASRModelLoadError, match="403 gated"):
        indic_backend.load_model(hf())


async def test_indic_inference_failure(fake_indic: dict[str, Any], tmp_path: Path) -> None:
    fake_indic["model"].fail = True

    with pytest.raises(TranscriptionError, match="onnx session crashed"):
        await IndicConformerBackend(hf()).transcribe(tmp_path / "a.wav", "or")


# --- real models (opt-in) ---------------------------------------------------------------

JFK_URL = "https://github.com/ggerganov/whisper.cpp/raw/master/samples/jfk.wav"  # public domain


def _fetch(url: str, dest: Path) -> Path:
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:  # noqa: S310 - fixed https URL
            dest.write_bytes(resp.read())
    except Exception as exc:  # pragma: no cover - network dependent
        pytest.skip(f"could not fetch sample audio: {exc}")
    return dest


def _has(module: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(module) is not None


async def _real_transcribe(backend: ASRBackend, path: Path, language: str) -> ASRResult:
    from app.services.audio.preprocessor import AudioPreprocessor

    processed = await AudioPreprocessor(backend.settings).process(uuid.uuid4(), path)
    return await backend.transcribe(processed.processed_path, language)


@pytest.mark.slow
@pytest.mark.skipif(not _has("faster_whisper"), reason="ml extras not installed")
async def test_real_whisper_english(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,
        storage_dir=tmp_path,
        whisper_model_size=os.environ.get("WHISPER_MODEL_SIZE", "base"),
    )
    audio = _fetch(JFK_URL, tmp_path / "jfk.wav")

    result = await _real_transcribe(WhisperBackend(settings), audio, "en")

    text = " ".join(s.text for s in result.segments).lower()
    assert "ask not what your country can do for you" in text
    assert all(w.start <= w.end for s in result.segments for w in s.words)


@pytest.mark.slow
@pytest.mark.skipif(
    not (_has("transformers") and _has("onnxruntime")), reason="indic extras not installed"
)
@pytest.mark.skipif(not os.environ.get("HF_TOKEN"), reason="HF_TOKEN not set")
@pytest.mark.skipif(
    not os.environ.get("POLYMOM_ODIA_SAMPLE_URL"),
    reason="set POLYMOM_ODIA_SAMPLE_URL to a public Odia clip",
)
async def test_real_indic_odia(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, storage_dir=tmp_path, hf_token=os.environ["HF_TOKEN"])
    audio = _fetch(os.environ["POLYMOM_ODIA_SAMPLE_URL"], tmp_path / "odia_sample")

    result = await _real_transcribe(IndicConformerBackend(settings), audio, "or")

    assert result.segments
    odia_chars = sum(0x0B00 <= ord(c) <= 0x0B7F for s in result.segments for c in s.text)
    assert odia_chars > 0  # native Odia script, not transliterated
