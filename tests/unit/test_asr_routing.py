from pathlib import Path

import pytest

from app.core.config import Settings
from app.core.exceptions import UnsupportedLanguageError
from app.schemas.asr import ASRResult
from app.services.asr.base import ASRBackend
from app.services.asr.indic_backend import IndicConformerBackend
from app.services.asr.mock_backend import MockASRBackend
from app.services.asr.router import ASRRouter
from app.services.asr.service import build_router, choose_language
from app.services.asr.whisper_backend import WhisperBackend


class Fake(ASRBackend):
    def __init__(self, name: str, languages: set[str], auto: bool = False) -> None:
        super().__init__(Settings(_env_file=None))
        self.name = name
        self._languages = frozenset(languages)
        self._auto = auto

    @property
    def model_name(self) -> str:
        return self.name

    @property
    def supported_languages(self) -> frozenset[str]:
        return self._languages

    @property
    def supports_auto_detect(self) -> bool:
        return self._auto

    async def transcribe(
        self, audio_path: Path, language: str | None, offset: float = 0.0
    ) -> ASRResult:
        raise NotImplementedError


@pytest.fixture
def router() -> ASRRouter:
    whisper = Fake("whisper", {"en", "hi"}, auto=True)
    indic = Fake("indic", {"or"})
    return ASRRouter(
        {"whisper": whisper, "indic": indic},
        {"en": "whisper", "hi": "whisper", "or": "indic", "ta": "indic"},
    )


@pytest.mark.parametrize(
    ("language", "backend"),
    [("en", "whisper"), ("hi", "whisper"), ("or", "indic"), ("OR", "indic"), (None, "whisper")],
)
def test_router_selects_backend_per_language(
    router: ASRRouter, language: str | None, backend: str
) -> None:
    assert router.select(language).name == backend


def test_router_rejects_unconfigured_language(router: ASRRouter) -> None:
    with pytest.raises(
        UnsupportedLanguageError, match="No ASR backend is configured for language 'fr'"
    ) as exc:
        router.select("fr")
    assert exc.value.details == {"language": "fr", "configured": ["en", "hi", "or", "ta"]}
    assert exc.value.status_code == 422


def test_router_rejects_language_the_backend_cannot_do(router: ASRRouter) -> None:
    with pytest.raises(UnsupportedLanguageError, match="does not support language 'ta'"):
        router.select("ta")


def test_router_auto_detect_requires_capable_backend() -> None:
    indic = Fake("indic", {"or"})
    router = ASRRouter({"indic": indic}, {"or": "indic"}, auto_backend="indic")

    with pytest.raises(UnsupportedLanguageError, match="cannot auto-detect"):
        router.select(None)


def test_router_validates_mapping_at_construction() -> None:
    with pytest.raises(ValueError, match="unknown backend"):
        ASRRouter({"whisper": Fake("whisper", {"en"}, True)}, {"or": "nemo"})


def test_router_languages(router: ASRRouter) -> None:
    assert router.languages == {"en", "hi", "or", "ta"}


def test_build_router_real_and_mock() -> None:
    real = build_router(Settings(_env_file=None))
    mock = build_router(Settings(_env_file=None, asr_backend="mock"))

    assert isinstance(real.select("en"), WhisperBackend)
    assert isinstance(real.select("or"), IndicConformerBackend)
    assert all(isinstance(mock.select(lang), MockASRBackend) for lang in ("en", "hi", "or", None))


def test_language_backends_config_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ASR_LANGUAGE_BACKENDS", "en:whisper, OR:Indic,")

    assert Settings(_env_file=None).asr_language_backends == {"en": "whisper", "or": "indic"}


def test_language_backends_config_rejects_garbage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ASR_LANGUAGE_BACKENDS", "en-whisper")

    with pytest.raises(ValueError, match="lang:backend"):
        Settings(_env_file=None)


@pytest.mark.parametrize(
    ("hints", "expected"), [([], None), (["hi"], "hi"), (["or"], "or"), (["en", "hi"], None)]
)
def test_choose_language(hints: list[str], expected: str | None) -> None:
    assert choose_language(hints) == expected
