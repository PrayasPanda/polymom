import json
import unicodedata
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from uuid import uuid4

import pytest
from httpx import AsyncClient

from app.core.config import Settings
from scripts import eval_asr
from tests.conftest import make_wav, requires_ffmpeg

pytestmark = requires_ffmpeg

URL = "/api/v1/meetings"


async def _process(client: AsyncClient, seconds: float = 9, **form: str | list[str]) -> str:
    response = await client.post(
        URL, files={"file": ("m.wav", make_wav(seconds=seconds))}, data=form
    )
    assert response.status_code == 202, response.text
    meeting_id: str = response.json()["meeting_id"]
    assert (await client.post(f"{URL}/{meeting_id}/process")).status_code == 202
    return meeting_id


async def test_transcript_json_is_nfc_native_script(client: AsyncClient) -> None:
    # 12 s, no hint: mock LID says en for 0-6 s and hi for 6-12 s; each region is routed.
    meeting_id = await _process(client, seconds=12)

    response = await client.get(f"{URL}/{meeting_id}/transcript")

    assert response.status_code == 200
    body = response.json()
    assert body["meeting_id"] == meeting_id
    assert [s["language"] for s in body["segments"]] == ["en", "en", "hi", "hi"]
    assert [s["id"] for s in body["segments"]] == [0, 1, 2, 3]
    for s in body["segments"]:
        assert unicodedata.is_normalized("NFC", s["text"])
        assert s["words"]
        assert s["backend"] == "mock-whisper"
        assert s["low_confidence"] is False
        assert s["lid_confidence"] is not None
    assert body["segments"][2]["text"].startswith("नमस्ते")
    assert {d["language"] for d in body["detected_languages"]} == {"en", "hi"}
    assert body["requested_language"] is None


async def test_single_language_hint_forces_language(client: AsyncClient) -> None:
    meeting_id = await _process(client, languages="or")

    body = (await client.get(f"{URL}/{meeting_id}/transcript")).json()

    assert {s["language"] for s in body["segments"]} == {"or"}
    assert body["requested_language"] == "or"


async def test_transcript_txt_and_srt(client: AsyncClient) -> None:
    meeting_id = await _process(client, seconds=6, languages="hi")

    txt = await client.get(f"{URL}/{meeting_id}/transcript", params={"format": "txt"})
    srt = await client.get(f"{URL}/{meeting_id}/transcript", params={"format": "srt"})

    assert txt.headers["content-type"].startswith("text/plain")
    assert txt.text.splitlines()[0] == "[00:00:00.000] (hi) नमस्ते सभी को, चलिए शुरू करते हैं।"
    assert srt.headers["content-type"].startswith("application/x-subrip")
    assert srt.headers["content-disposition"] == f'attachment; filename="{meeting_id}.srt"'
    assert srt.text.startswith("1\n00:00:00,000 --> 00:00:03,000\nनमस्ते")
    assert "\n2\n00:00:03,000 --> 00:00:06,000\n" in srt.text


async def test_transcript_rejects_unknown_format(client: AsyncClient) -> None:
    meeting_id = await _process(client, seconds=3)

    response = await client.get(f"{URL}/{meeting_id}/transcript", params={"format": "docx"})

    assert response.status_code == 422


async def test_transcript_before_processing_is_409_and_unknown_is_404(client: AsyncClient) -> None:
    upload = await client.post(URL, files={"file": ("m.wav", make_wav(seconds=3))})
    meeting_id = upload.json()["meeting_id"]

    early = await client.get(f"{URL}/{meeting_id}/transcript")

    assert early.status_code == 409
    assert early.json()["error"]["code"] == "transcript_not_available"
    assert (await client.get(f"{URL}/{uuid4()}/transcript")).status_code == 404


async def test_asr_failure_keeps_earlier_stage_outputs(
    tmp_path: Path, client_factory: Callable[[Settings], AsyncIterator[AsyncClient]]
) -> None:
    settings = Settings(
        _env_file=None,
        app_env="test",
        storage_dir=tmp_path,
        diarization_backend="mock",
        asr_backend="real",
        lid_backend="mock",
        asr_language_backends={"en": "whisper", "hi": "whisper", "or": "indic"},
    )
    async for client in client_factory(settings):
        meeting_id = await _process(client, seconds=4, languages="or")

        body = (await client.get(f"{URL}/{meeting_id}")).json()

        assert body["status"] == "failed"
        assert body["error"].startswith("asr_model_unavailable:")
        assert body["audio_quality"] is not None
        speakers = await client.get(f"{URL}/{meeting_id}/speakers")
        assert speakers.status_code == 200  # diarization survived the ASR failure
        assert (await client.get(f"{URL}/{meeting_id}/transcript")).status_code == 409


def test_eval_script_end_to_end_with_mock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)  # keep the script from reading the repo's .env
    data = tmp_path / "data" / "hi"
    data.mkdir(parents=True)
    (data / "clip.wav").write_bytes(make_wav(seconds=3))
    (data / "clip.txt").write_text("नमस्ते सभी को, चलिए शुरू करते हैं।", encoding="utf-8")
    report = tmp_path / "report.json"

    assert eval_asr.main([str(tmp_path / "data"), "--backend", "mock", "--json", str(report)]) == 0

    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["languages"] == [{"language": "hi", "files": 1, "wer": 0.0, "cer": 0.0}]
