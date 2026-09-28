import pytest

from app.core.config import Settings


def test_allowed_extensions_parsed_from_csv(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALLOWED_EXTENSIONS", "wav, .MP3,,m4a")

    settings = Settings(_env_file=None)

    assert settings.allowed_extensions == frozenset({"wav", "mp3", "m4a"})


def test_max_upload_bytes() -> None:
    assert Settings(_env_file=None, max_upload_mb=2).max_upload_bytes == 2 * 1024 * 1024
