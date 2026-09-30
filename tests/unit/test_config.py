import re
from pathlib import Path

import pytest

from app.core.config import Settings


def test_allowed_extensions_parsed_from_csv(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALLOWED_EXTENSIONS", "wav, .MP3,,m4a")

    settings = Settings(_env_file=None)

    assert settings.allowed_extensions == frozenset({"wav", "mp3", "m4a"})


def test_max_upload_bytes() -> None:
    assert Settings(_env_file=None, max_upload_mb=2).max_upload_bytes == 2 * 1024 * 1024


def test_env_example_documents_every_setting() -> None:
    """.env.example and Settings must list the same keys (OTel SDK vars excepted)."""
    text = (Path(__file__).parents[2] / ".env.example").read_text(encoding="utf-8")
    documented = set(re.findall(r"^#? ?([A-Z][A-Z0-9_]+)=", text, re.MULTILINE))
    fields = {name.upper() for name in Settings.model_fields}
    assert fields - documented == set()
    assert {k for k in documented - fields if not k.startswith("OTEL_EXPORTER")} == set()
