from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from httpx import AsyncClient

from app.core.config import Settings
from tests.conftest import make_wav, requires_ffmpeg

pytestmark = requires_ffmpeg

URL = "/api/v1/meetings"


async def upload(
    client: AsyncClient, data: bytes, name: str = "call.wav", **form: Any
) -> dict[str, Any]:
    response = await client.post(URL, files={"file": (name, data, "audio/wav")}, data=form)
    return {"status": response.status_code, "json": response.json()}


async def test_upload_returns_202_and_stores_file(
    client: AsyncClient, settings: Settings, wav_bytes: bytes
) -> None:
    result = await upload(
        client, wav_bytes, title="Standup", expected_speakers="3", languages=["en", "hi,or"]
    )

    assert result["status"] == 202
    body = result["json"]
    assert body["status"] == "queued"
    assert set(body) == {"meeting_id", "status", "created_at", "duplicate"}
    assert body["duplicate"] is False
    stored = settings.storage_dir / "meetings" / body["meeting_id"] / "upload" / "original.wav"
    assert stored.read_bytes() == wav_bytes


async def test_get_meeting_returns_metadata(client: AsyncClient, wav_bytes: bytes) -> None:
    created = (await upload(client, wav_bytes, title="Standup", languages="en,hi"))["json"]

    response = await client.get(f"{URL}/{created['meeting_id']}")

    assert response.status_code == 200
    body = response.json()
    assert body["meeting_id"] == created["meeting_id"]
    assert body["title"] == "Standup"
    assert body["original_filename"] == "call.wav"
    assert body["mime_type"] == "audio/x-wav"
    assert body["size_bytes"] == len(wav_bytes)
    assert body["languages_hint"] == ["en", "hi"]
    assert body["expected_speakers"] is None
    assert body["status"] == "queued"
    assert body["audio_metadata"]["sample_rate"] == 16000
    assert body["duration_seconds"] == pytest.approx(0.5, abs=0.01)
    assert "stored_path" not in body


async def test_list_is_paginated_newest_first(client: AsyncClient, wav_bytes: bytes) -> None:
    ids = [
        (await upload(client, wav_bytes + bytes(i), title=f"m{i}"))["json"]["meeting_id"]
        for i in range(3)
    ]

    first = (await client.get(URL, params={"limit": 2})).json()
    second = (await client.get(URL, params={"limit": 2, "cursor": first["next_cursor"]})).json()

    assert first["total"] == 3
    assert [m["meeting_id"] for m in first["items"]] == ids[::-1][:2]
    assert [m["meeting_id"] for m in second["items"]] == ids[:1]
    assert first["limit"] == 2
    assert second["next_cursor"] is None


async def test_delete_removes_record_and_file(
    client: AsyncClient, settings: Settings, wav_bytes: bytes
) -> None:
    meeting_id = (await upload(client, wav_bytes))["json"]["meeting_id"]
    stored = settings.storage_dir / "meetings" / meeting_id / "upload" / "original.wav"
    assert stored.exists()

    response = await client.delete(f"{URL}/{meeting_id}")

    assert response.status_code == 204
    assert not stored.exists()
    assert (await client.get(f"{URL}/{meeting_id}")).status_code == 404


@pytest.mark.parametrize("method", ["get", "delete"])
async def test_unknown_meeting_returns_404_envelope(client: AsyncClient, method: str) -> None:
    meeting_id = str(uuid4())

    response = await client.request(method.upper(), f"{URL}/{meeting_id}")

    assert response.status_code == 404
    error = response.json()["error"]
    assert error.pop("remediation").startswith("Check the id")
    assert len(error.pop("request_id")) == 32
    assert response.headers["x-request-id"]
    assert {"error": error} == {
        "error": {
            "code": "meeting_not_found",
            "message": f"Meeting {meeting_id} not found.",
            "details": {"meeting_id": meeting_id},
        }
    }


async def test_malformed_meeting_id_is_422(client: AsyncClient) -> None:
    response = await client.get(f"{URL}/not-a-uuid")

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


async def test_fake_extension_returns_415_and_leaves_no_files(
    client: AsyncClient, settings: Settings
) -> None:
    result = await upload(client, b"plain text pretending to be audio" * 10, "fake.wav")

    assert result["status"] == 415
    assert result["json"]["error"]["code"] == "unsupported_file_type"
    assert not any((settings.storage_dir / "tmp").iterdir())
    assert not (settings.storage_dir / "meetings").exists()
    assert (await client.get(URL)).json()["total"] == 0


async def test_video_without_audio_returns_422(
    client: AsyncClient, video_without_audio_bytes: bytes
) -> None:
    result = await upload(client, video_without_audio_bytes, "screen.mp4")

    assert result["status"] == 422
    error = result["json"]["error"]
    assert error.pop("remediation")
    assert error.pop("request_id")
    assert error == {
        "code": "corrupted_media",
        "message": "The file contains no audio stream.",
        "details": {"reason": "no_audio_stream"},
    }


async def test_empty_upload_returns_422(client: AsyncClient) -> None:
    result = await upload(client, b"", "empty.wav")

    assert result["status"] == 422
    assert result["json"]["error"]["code"] == "empty_file"


@pytest.mark.parametrize(
    ("form", "field"),
    [
        ({"expected_speakers": "0"}, "expected_speakers"),
        ({"expected_speakers": "21"}, "expected_speakers"),
        ({"title": "x" * 201}, "title"),
    ],
)
async def test_invalid_form_fields_return_422(
    client: AsyncClient, wav_bytes: bytes, form: dict[str, str], field: str
) -> None:
    result = await upload(client, wav_bytes, **form)

    assert result["status"] == 422
    errors = result["json"]["error"]["details"]["errors"]
    assert any(field in e["loc"] for e in errors)


async def test_unsupported_language_returns_422(client: AsyncClient, wav_bytes: bytes) -> None:
    result = await upload(client, wav_bytes, languages="en,fr")

    assert result["status"] == 422
    assert result["json"]["error"]["details"]["invalid"] == ["fr"]


async def test_missing_file_returns_422(client: AsyncClient) -> None:
    response = await client.post(URL, data={"title": "no file"})

    assert response.status_code == 422


async def test_oversized_upload_returns_413(
    tmp_path: Path, client_factory: Callable[[Settings], AsyncIterator[AsyncClient]]
) -> None:
    settings = Settings(
        _env_file=None,
        app_env="test",
        storage_dir=tmp_path,
        max_upload_mb=1,
        api_key_required=False,
        pipeline_execution="inline",
    )

    async for client in client_factory(settings):
        result = await upload(client, make_wav(seconds=40), "big.wav")

        assert result["status"] == 413
        assert result["json"]["error"]["code"] == "file_too_large"
        assert not any((settings.storage_dir / "tmp").glob("*"))
        assert not (settings.storage_dir / "meetings").exists()


async def test_openapi_documents_error_responses(client: AsyncClient) -> None:
    spec = (await client.get("/openapi.json")).json()
    post = spec["paths"][URL]["post"]

    assert {"202", "413", "415", "422"} <= set(post["responses"])
    assert "ErrorResponse" in spec["components"]["schemas"]
