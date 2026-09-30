import importlib.util
import os
import struct
import sys
import types
import uuid
import wave
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from app.core.config import Settings
from app.core.exceptions import DiarizationModelLoadError, ResourceExhaustedError
from app.services.diarization import pyannote_backend
from app.services.diarization.mock_backend import MockDiarizationBackend, speaker_embedding
from app.services.diarization.postprocess import RawDiarization
from app.services.diarization.pyannote_backend import PyannoteDiarizationBackend, to_raw
from app.services.diarization.service import DiarizationService, build_backend
from tests.conftest import make_wav


def write_wav(path: Path, data: bytes) -> Path:
    path.write_bytes(data)
    return path


def silent_wav(path: Path, seconds: float = 3.0, rate: int = 16000) -> Path:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(struct.pack("<h", 0) * int(seconds * rate))
    return path


@pytest.fixture
def mock_settings(tmp_path: Path) -> Settings:
    return Settings(_env_file=None, storage_dir=tmp_path, diarization_backend="mock")


# --- mock backend -------------------------------------------------------------


async def test_mock_backend_alternates_speakers(mock_settings: Settings, tmp_path: Path) -> None:
    audio = write_wav(tmp_path / "a.wav", make_wav(seconds=9))

    result = await MockDiarizationBackend(mock_settings).diarize(audio, num_speakers=3)

    assert result.num_speakers == 3
    assert result.model_name == "mock"
    assert [t.speaker_label for t in result.turns] == [
        "Person 1",
        "Person 2",
        "Person 3",
        "Person 1",
        "Person 2",
    ]
    assert result.turns[-1].end == 9.0


async def test_mock_backend_no_speech(mock_settings: Settings, tmp_path: Path) -> None:
    result = await MockDiarizationBackend(mock_settings).diarize(silent_wav(tmp_path / "s.wav"))

    assert (result.num_speakers, result.turns) == (0, [])


async def test_mock_backend_uses_max_speakers_hint(mock_settings: Settings, tmp_path: Path) -> None:
    audio = write_wav(tmp_path / "a.wav", make_wav(seconds=9))

    raw = await MockDiarizationBackend(mock_settings).diarize_raw(audio, max_speakers=4)

    assert set(raw.embeddings) == {"SPEAKER_00", "SPEAKER_01", "SPEAKER_02", "SPEAKER_03"}
    assert raw.embeddings["SPEAKER_01"] == speaker_embedding(1)


def test_build_backend_selects_by_config() -> None:
    assert isinstance(
        build_backend(Settings(_env_file=None, diarization_backend="mock")), MockDiarizationBackend
    )
    assert isinstance(build_backend(Settings(_env_file=None)), PyannoteDiarizationBackend)


# --- service: chunking for long recordings -------------------------------------


async def test_service_chunks_long_audio_and_keeps_labels_consistent(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,
        storage_dir=tmp_path,
        diarization_backend="mock",
        diarization_chunk_threshold_seconds=30,
        chunk_length_seconds=20,
        chunk_overlap_seconds=2,
    )
    audio = write_wav(tmp_path / "long.wav", make_wav(seconds=60))
    service = DiarizationService(MockDiarizationBackend(settings), settings)

    result = await service.diarize(uuid.uuid4(), audio, num_speakers=2)

    assert result.num_speakers == 2
    assert {t.speaker_label for t in result.turns} == {"Person 1", "Person 2"}
    assert result.turns[0].start == 0
    assert result.turns[-1].end == 60
    # Chunk overlap windows are de-duplicated: the same speaker never overlaps itself.
    for a, b in zip(result.turns, result.turns[1:], strict=False):
        assert b.start >= a.start
    assert not [p for p in tmp_path.iterdir() if p.name.endswith("_diarization_chunks")]


async def test_service_does_not_chunk_short_audio(mock_settings: Settings, tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []

    class Spy(MockDiarizationBackend):
        async def diarize_raw(
            self,
            audio_path: Path,
            num_speakers: int | None = None,
            min_speakers: int | None = None,
            max_speakers: int | None = None,
        ) -> RawDiarization:
            calls.append({"num": num_speakers, "max": max_speakers})
            return await super().diarize_raw(audio_path, num_speakers, min_speakers, max_speakers)

    audio = write_wav(tmp_path / "a.wav", make_wav(seconds=4))

    await DiarizationService(Spy(mock_settings), mock_settings).diarize(uuid.uuid4(), audio, 2)

    assert calls == [{"num": 2, "max": None}]


# --- pyannote backend, with torch/pyannote faked -------------------------------


class FakeTurn:
    def __init__(self, start: float, end: float) -> None:
        self.start, self.end = start, end


class FakeAnnotation:
    def __init__(self, tracks: list[tuple[float, float, str]]) -> None:
        self._tracks = tracks

    def itertracks(self, yield_label: bool = False) -> Iterator[tuple[FakeTurn, str, str]]:
        for i, (start, end, label) in enumerate(self._tracks):
            yield FakeTurn(start, end), f"t{i}", label

    def labels(self) -> list[str]:
        return sorted({label for _, _, label in self._tracks})


class FakePipeline:
    def __init__(self, fail: bool = False) -> None:
        self.device: Any = None
        self.calls: list[dict[str, Any]] = []
        self.fail = fail

    def to(self, device: Any) -> "FakePipeline":
        self.device = device
        return self

    def __call__(self, audio: Any, **kwargs: Any) -> tuple[FakeAnnotation, list[list[float]]]:
        if self.fail:
            raise RuntimeError("CUDA out of memory")
        self.calls.append(kwargs)
        annotation = FakeAnnotation([(0.0, 2.0, "SPEAKER_01"), (2.5, 4.0, "SPEAKER_00")])
        return annotation, [[0.0, 1.0], [1.0, 0.0]]


@pytest.fixture
def fake_ml(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Install fake ``torch`` and ``pyannote.audio`` modules."""
    state: dict[str, Any] = {"cuda": False, "pipeline": FakePipeline(), "loads": 0, "raise": None}

    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: state["cuda"])  # type: ignore[attr-defined]
    torch.device = lambda name: f"device:{name}"  # type: ignore[attr-defined]

    def from_pretrained(model: str, use_auth_token: str) -> Any:
        state["loads"] += 1
        state["token"] = use_auth_token
        if state["raise"]:
            raise state["raise"]
        return state["pipeline"]

    audio = types.ModuleType("pyannote.audio")
    audio.Pipeline = types.SimpleNamespace(from_pretrained=from_pretrained)  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "pyannote.audio", audio)
    monkeypatch.setattr(pyannote_backend, "_pipelines", {})
    monkeypatch.setattr(pyannote_backend, "load_waveform", lambda path: {"waveform": path})
    return state


def hf_settings(**kwargs: Any) -> Settings:
    return Settings(_env_file=None, hf_token="hf_test", **kwargs)


async def test_pyannote_backend_end_to_end_with_fakes(
    fake_ml: dict[str, Any], tmp_path: Path
) -> None:
    backend = PyannoteDiarizationBackend(hf_settings(merge_gap_seconds=0.1))

    result = await backend.diarize(tmp_path / "a.wav", num_speakers=2)
    await backend.diarize(tmp_path / "a.wav", min_speakers=1, max_speakers=3)

    assert result.model_name == "pyannote/speaker-diarization-3.1"
    assert [(t.speaker_label, t.raw_label) for t in result.turns] == [
        ("Person 1", "SPEAKER_01"),
        ("Person 2", "SPEAKER_00"),
    ]
    pipeline: FakePipeline = fake_ml["pipeline"]
    assert pipeline.calls == [
        {"return_embeddings": True, "num_speakers": 2},
        {"return_embeddings": True, "min_speakers": 1, "max_speakers": 3},
    ]
    assert fake_ml["loads"] == 1  # loaded once, then cached
    assert fake_ml["token"] == "hf_test"
    assert pipeline.device == "device:cpu"


async def test_pyannote_embeddings_follow_label_order(
    fake_ml: dict[str, Any], tmp_path: Path
) -> None:
    raw = await PyannoteDiarizationBackend(hf_settings()).diarize_raw(tmp_path / "a.wav")

    assert raw.embeddings == {"SPEAKER_00": [0.0, 1.0], "SPEAKER_01": [1.0, 0.0]}


def test_device_auto_prefers_cuda(fake_ml: dict[str, Any]) -> None:
    fake_ml["cuda"] = True

    pyannote_backend.load_pipeline(hf_settings(device="auto"))

    assert fake_ml["pipeline"].device == "device:cuda"


def test_device_cuda_unavailable(fake_ml: dict[str, Any]) -> None:
    with pytest.raises(DiarizationModelLoadError, match="no CUDA device"):
        pyannote_backend.load_pipeline(hf_settings(device="cuda"))


def test_missing_token(fake_ml: dict[str, Any]) -> None:
    with pytest.raises(DiarizationModelLoadError, match="HF_TOKEN is not set") as exc:
        pyannote_backend.load_pipeline(Settings(_env_file=None))
    assert "huggingface.co/pyannote/speaker-diarization-3.1" in exc.value.message
    assert exc.value.details == {"reason": "missing_hf_token"}


def test_gated_model_access_denied(fake_ml: dict[str, Any]) -> None:
    fake_ml["pipeline"] = None

    with pytest.raises(DiarizationModelLoadError, match="denied") as exc:
        pyannote_backend.load_pipeline(hf_settings())
    assert exc.value.status_code == 503


def test_download_failure(fake_ml: dict[str, Any]) -> None:
    fake_ml["raise"] = OSError("connection reset")

    with pytest.raises(DiarizationModelLoadError, match="connection reset"):
        pyannote_backend.load_pipeline(hf_settings())


def test_ml_extras_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)

    with pytest.raises(DiarizationModelLoadError, match="uv sync --extra ml"):
        pyannote_backend.load_pipeline(hf_settings())


async def test_cuda_oom_falls_back_to_cpu_then_reports_exhaustion(
    fake_ml: dict[str, Any], tmp_path: Path
) -> None:
    fake_ml["pipeline"] = FakePipeline(fail=True)  # raises "CUDA out of memory" on GPU and CPU

    with pytest.raises(ResourceExhaustedError, match="GPU and CPU") as exc:
        await PyannoteDiarizationBackend(hf_settings()).diarize(tmp_path / "a.wav")
    assert exc.value.code == "resource_exhausted"


def test_to_raw_without_embeddings() -> None:
    raw = to_raw(FakeAnnotation([(0.0, 1.0, "SPEAKER_00")]), None)

    assert raw.embeddings == {}
    assert [(s.start, s.end, s.label) for s in raw.segments] == [(0.0, 1.0, "SPEAKER_00")]


# --- real model (opt-in) -----------------------------------------------------------

HAS_ML = (
    importlib.util.find_spec("pyannote") is not None
    and importlib.util.find_spec("torch") is not None
)


@pytest.mark.slow
@pytest.mark.skipif(not HAS_ML, reason="ml extras not installed (uv sync --extra ml)")
@pytest.mark.skipif(not os.environ.get("HF_TOKEN"), reason="HF_TOKEN not set")
async def test_real_pyannote_model(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, hf_token=os.environ["HF_TOKEN"], device="auto")
    audio = write_wav(tmp_path / "a.wav", make_wav(seconds=6))

    result = await PyannoteDiarizationBackend(settings).diarize(audio)

    assert result.model_name == "pyannote/speaker-diarization-3.1"
    assert result.turns == sorted(result.turns, key=lambda t: t.start)
    assert result.num_speakers == len({t.speaker_label for t in result.turns})
