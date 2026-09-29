from collections.abc import AsyncIterator, Callable
from uuid import uuid4

from httpx import AsyncClient

from app.core.config import Settings
from tests.conftest import make_wav, requires_ffmpeg

pytestmark = requires_ffmpeg

URL = "/api/v1/meetings"


async def _process(client: AsyncClient, seconds: float = 18) -> str:
    upload = await client.post(URL, files={"file": ("m.wav", make_wav(seconds=seconds))})
    meeting_id: str = upload.json()["meeting_id"]
    await client.post(f"{URL}/{meeting_id}/process")
    return meeting_id


async def test_summary_json_md_and_regenerate(client: AsyncClient) -> None:
    # Mock LLM turns each speaker's first utterance into a grounded key point.
    meeting_id = await _process(client)
    assert (await client.get(f"{URL}/{meeting_id}")).json()["status"] == "completed"
    await client.patch(f"{URL}/{meeting_id}/speakers", json={"names": {"Person 1": "Ravi"}})

    body = (await client.get(f"{URL}/{meeting_id}/summary")).json()

    assert body["meeting_id"] == meeting_id
    assert body["speaker_names"] == {"Person 1": "Ravi"}
    assert {k["speakers_involved"][0] for k in body["key_points"]} == {"Person 1", "Person 2"}
    assert all(e["start"] is not None for k in body["key_points"] for e in k["evidence"])
    assert body["model_info"]["provider"] == "mock"
    assert body["model_info"]["strategy"] == "single_pass"
    assert body["source_languages"] == ["en", "hi", "or"]
    report = body["verification_report"]
    assert report["dropped"] == 0
    assert report["evidence_passed"] == report["evidence_checked"]

    md = await client.get(f"{URL}/{meeting_id}/summary", params={"format": "md"})
    assert md.headers["content-type"].startswith("text/markdown")
    assert "## Key discussion points" in md.text
    assert "Ravi" in md.text

    regen = await client.post(
        f"{URL}/{meeting_id}/summary/regenerate", json={"output_language": "hi", "model": "mock-2"}
    )
    assert regen.status_code == 202
    body = (await client.get(f"{URL}/{meeting_id}/summary")).json()
    assert (body["output_language"], body["model_info"]["model"]) == ("hi", "mock-2")
    assert (await client.post(f"{URL}/{meeting_id}/summary/regenerate")).status_code == 202


async def test_summary_errors(client: AsyncClient) -> None:
    upload = await client.post(URL, files={"file": ("m.wav", make_wav(seconds=3))})
    meeting_id = upload.json()["meeting_id"]

    early = await client.get(f"{URL}/{meeting_id}/summary")
    assert early.status_code == 409
    assert early.json()["error"]["code"] == "summary_not_available"
    regen = await client.post(f"{URL}/{meeting_id}/summary/regenerate")
    assert regen.status_code == 409
    assert (await client.get(f"{URL}/{uuid4()}/summary")).status_code == 404
    assert (await client.post(f"{URL}/{uuid4()}/summary/regenerate")).status_code == 404


async def test_llm_failure_keeps_transcript_and_analytics(
    settings: Settings,
    client_factory: Callable[[Settings], AsyncIterator[AsyncClient]],
) -> None:
    broken = settings.model_copy(update={"llm_provider": "openai", "llm_api_key": None})
    async for client in client_factory(broken):
        meeting_id = await _process(client, 6)

        meeting = (await client.get(f"{URL}/{meeting_id}")).json()
        assert meeting["status"] == "completed_with_errors"
        assert meeting["error"].startswith("summarize: llm_error: LLM_API_KEY is required")
        summary = await client.get(f"{URL}/{meeting_id}/summary")
        assert summary.status_code == 409
        assert summary.json()["error"]["details"]["summary_error"].startswith("llm_error")
        assert (await client.get(f"{URL}/{meeting_id}/transcript")).status_code == 200
        assert (await client.get(f"{URL}/{meeting_id}/analytics")).status_code == 200

        # a failed regeneration keeps the meeting in completed_with_errors
        assert (await client.post(f"{URL}/{meeting_id}/summary/regenerate")).status_code == 202
        meeting = (await client.get(f"{URL}/{meeting_id}")).json()
        assert meeting["status"] == "completed_with_errors"
        # reprocessing needs force, like a completed meeting
        assert (await client.post(f"{URL}/{meeting_id}/process")).status_code == 409


async def test_regenerate_recovers_from_failed_summary(
    settings: Settings,
    client_factory: Callable[[Settings], AsyncIterator[AsyncClient]],
) -> None:
    broken = settings.model_copy(update={"llm_provider": "openai", "llm_api_key": None})
    async for client in client_factory(broken):
        meeting_id = await _process(client, 6)
    # same database, now with a working provider
    async for client in client_factory(settings):
        assert (await client.post(f"{URL}/{meeting_id}/summary/regenerate")).status_code == 202
        meeting = (await client.get(f"{URL}/{meeting_id}")).json()
        assert (meeting["status"], meeting["error"]) == ("completed", None)
        assert (await client.get(f"{URL}/{meeting_id}/summary")).status_code == 200
