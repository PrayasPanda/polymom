import struct
import wave
from collections.abc import AsyncIterator, Callable
from io import BytesIO
from pathlib import Path
from uuid import uuid4

from httpx import AsyncClient

from app.core.config import Settings
from tests.conftest import make_wav, requires_ffmpeg

pytestmark = requires_ffmpeg

URL = "/api/v1/meetings"


async def _upload(client: AsyncClient, data: bytes, **form: str) -> str:
    response = await client.post(URL, files={"file": ("m.wav", data)}, data=form)
    assert response.status_code == 202, response.text
    meeting_id: str = response.json()["meeting_id"]
    return meeting_id


async def test_speakers_after_processing(client: AsyncClient, settings: Settings) -> None:
    meeting_id = await _upload(client, make_wav(seconds=8))

    assert (await client.post(f"{URL}/{meeting_id}/process")).status_code == 202

    assert (await client.get(f"{URL}/{meeting_id}")).json()["status"] == "completed"
    response = await client.get(f"{URL}/{meeting_id}/speakers")
    assert response.status_code == 200
    body = response.json()
    assert body["meeting_id"] == meeting_id
    assert body["num_speakers"] == 2
    assert body["speakers"] == ["Person 1", "Person 2"]
    assert body["model_name"] == "mock"
    assert body["turns"][0]["speaker_label"] == "Person 1"
    assert body["turns"][0]["start"] == 0
    assert [t["start"] for t in body["turns"]] == sorted(t["start"] for t in body["turns"])
    assert set(body["turns"][0]) == {
        "speaker_label",
        "raw_label",
        "start",
        "end",
        "duration",
        "is_overlap",
    }


async def test_expected_speakers_is_passed_to_diarization(client: AsyncClient) -> None:
    meeting_id = await _upload(client, make_wav(seconds=8), expected_speakers="3")
    await client.post(f"{URL}/{meeting_id}/process")

    body = (await client.get(f"{URL}/{meeting_id}/speakers")).json()

    assert body["num_speakers"] == 3


async def test_audio_without_speech_completes_with_zero_speakers(client: AsyncClient) -> None:
    buf = BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(struct.pack("<h", 0) * 16000 * 3)
    meeting_id = await _upload(client, buf.getvalue())

    await client.post(f"{URL}/{meeting_id}/process")

    assert (await client.get(f"{URL}/{meeting_id}")).json()["status"] == "completed"
    body = (await client.get(f"{URL}/{meeting_id}/speakers")).json()
    assert (body["num_speakers"], body["speakers"], body["turns"]) == (0, [], [])


async def test_speakers_before_processing_is_409(client: AsyncClient) -> None:
    meeting_id = await _upload(client, make_wav(seconds=3))

    response = await client.get(f"{URL}/{meeting_id}/speakers")

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "diarization_not_available"
    assert response.json()["error"]["details"]["status"] == "queued"


async def test_speakers_unknown_meeting_is_404(client: AsyncClient) -> None:
    assert (await client.get(f"{URL}/{uuid4()}/speakers")).status_code == 404


async def test_missing_model_marks_meeting_failed_with_actionable_error(
    tmp_path: Path,
    client_factory: Callable[[Settings], AsyncIterator[AsyncClient]],
) -> None:
    settings = Settings(
        _env_file=None,
        app_env="test",
        api_key_required=False,
        pipeline_execution="inline",
        storage_dir=tmp_path,
        diarization_backend="pyannote",
        asr_backend="mock",
        lid_backend="mock",
    )
    async for client in client_factory(settings):
        meeting_id = await _upload(client, make_wav(seconds=3))
        await client.post(f"{URL}/{meeting_id}/process")

        body = (await client.get(f"{URL}/{meeting_id}")).json()

        assert body["status"] == "failed"
        assert body["error"].startswith("diarization_model_unavailable: HF_TOKEN is not set")
        assert body["audio_quality"] is not None  # preprocess still succeeded
