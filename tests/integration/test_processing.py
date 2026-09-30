from collections.abc import AsyncIterator, Callable
from pathlib import Path
from uuid import uuid4

from httpx import AsyncClient

from app.core.config import Settings
from tests.conftest import requires_ffmpeg

pytestmark = requires_ffmpeg

URL = "/api/v1/meetings"


async def _upload(client: AsyncClient, path: Path) -> str:
    response = await client.post(URL, files={"file": (path.name, path.read_bytes())})
    assert response.status_code == 202, response.text
    meeting_id: str = response.json()["meeting_id"]
    return meeting_id


async def test_process_completes_and_populates_audio_quality(
    client: AsyncClient, settings: Settings, stereo_44k_path: Path
) -> None:
    meeting_id = await _upload(client, stereo_44k_path)

    response = await client.post(f"{URL}/{meeting_id}/process")

    assert response.status_code == 202
    assert response.json() == {"meeting_id": meeting_id, "status": "processing"}
    # httpx's ASGI transport returns after background tasks have finished.
    body = (await client.get(f"{URL}/{meeting_id}")).json()
    assert body["status"] == "completed"
    assert body["error"] is None
    quality = body["audio_quality"]
    assert (quality["sample_rate"], quality["channels"]) == (16000, 1)
    assert quality["duration_seconds"] > 2.9
    assert quality["warnings"] == []
    assert "processed_path" not in quality
    run_id = (await client.get(f"{URL}/{meeting_id}/runs")).json()["items"][0]["run_id"]
    assert (settings.storage_dir / "meetings" / meeting_id / run_id / "processed.wav").exists()
    assert not (settings.processed_dir / f"{meeting_id}.wav").exists()  # work copy removed


async def test_process_rejects_completed_unless_forced(
    client: AsyncClient, stereo_44k_path: Path
) -> None:
    meeting_id = await _upload(client, stereo_44k_path)
    await client.post(f"{URL}/{meeting_id}/process")

    again = await client.post(f"{URL}/{meeting_id}/process")
    forced = await client.post(f"{URL}/{meeting_id}/process", params={"force": "true"})

    assert again.status_code == 409
    assert again.json()["error"]["code"] == "meeting_state_conflict"
    assert again.json()["error"]["details"]["status"] == "completed"
    assert forced.status_code == 202
    assert (await client.get(f"{URL}/{meeting_id}")).json()["status"] == "completed"


async def test_process_unknown_meeting_is_404(client: AsyncClient) -> None:
    response = await client.post(f"{URL}/{uuid4()}/process")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "meeting_not_found"


async def test_failed_processing_records_typed_error(
    tmp_path: Path,
    stereo_44k_path: Path,
    client_factory: Callable[[Settings], AsyncIterator[AsyncClient]],
) -> None:
    settings = Settings(
        _env_file=None,
        app_env="test",
        api_key_required=False,
        pipeline_execution="inline",
        storage_dir=tmp_path,
        ffmpeg_path="no-such-ffmpeg-xyz",
        diarization_backend="mock",
        asr_backend="mock",
        lid_backend="mock",
    )
    async for client in client_factory(settings):
        meeting_id = await _upload(client, stereo_44k_path)

        assert (await client.post(f"{URL}/{meeting_id}/process")).status_code == 202

        body = (await client.get(f"{URL}/{meeting_id}")).json()
        assert body["status"] == "failed"
        assert body["error"].startswith("audio_processing_failed:")
        assert body["audio_quality"] is None
        # A failed meeting can be retried without force.
        assert (await client.post(f"{URL}/{meeting_id}/process")).status_code == 202


async def test_delete_removes_processed_file(
    client: AsyncClient, settings: Settings, stereo_44k_path: Path
) -> None:
    meeting_id = await _upload(client, stereo_44k_path)
    await client.post(f"{URL}/{meeting_id}/process")
    run_id = (await client.get(f"{URL}/{meeting_id}/runs")).json()["items"][0]["run_id"]
    processed = settings.storage_dir / "meetings" / meeting_id / run_id / "processed.wav"
    assert processed.exists()

    assert (await client.delete(f"{URL}/{meeting_id}")).status_code == 204
    assert not processed.exists()
