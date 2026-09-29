from uuid import uuid4

from httpx import AsyncClient

from tests.conftest import make_wav, requires_ffmpeg

pytestmark = requires_ffmpeg

URL = "/api/v1/meetings"


async def _process(client: AsyncClient, seconds: float, **form: str) -> str:
    upload = await client.post(URL, files={"file": ("m.wav", make_wav(seconds=seconds))}, data=form)
    meeting_id: str = upload.json()["meeting_id"]
    assert (await client.post(f"{URL}/{meeting_id}/process")).status_code == 202
    assert (await client.get(f"{URL}/{meeting_id}")).json()["status"] == "completed"
    return meeting_id


async def test_code_switching_meeting_is_routed_per_language(client: AsyncClient) -> None:
    # Mock diarization: speakers alternate every 2 s. Mock LID: en 0-6 s, hi 6-12 s, or 12-18 s.
    meeting_id = await _process(client, 18)

    transcript = (await client.get(f"{URL}/{meeting_id}/transcript")).json()
    segments = transcript["segments"]
    assert [(s["start"], s["language"], s["backend"]) for s in segments] == [
        (0, "en", "mock-whisper"),
        (3, "en", "mock-whisper"),
        (6, "hi", "mock-whisper"),
        (9, "hi", "mock-whisper"),
        (12, "or", "mock-indic"),
        (15, "or", "mock-indic"),
    ]
    assert [s["start"] for s in segments] == sorted(s["start"] for s in segments)
    assert all(s["primary_language"] == s["language"] for s in segments)
    assert all(w["script"] for s in segments for w in s["words"] if w["language"])

    languages = await client.get(f"{URL}/{meeting_id}/languages")
    assert languages.status_code == 200
    body = languages.json()
    assert body["meeting_id"] == meeting_id
    assert [(x["language"], x["duration_seconds"], x["percentage"]) for x in body["languages"]] == [
        ("en", 6, 33.33),
        ("hi", 6, 33.33),
        ("or", 6, 33.33),
    ]
    assert [
        (p["timestamp"], p["from_language"], p["to_language"]) for p in body["switch_points"]
    ] == [
        (6, "en", "hi"),
        (12, "hi", "or"),
    ]
    assert body["num_switches"] == 2
    assert {s["speaker"] for s in body["speakers"]} == {"Person 1", "Person 2"}
    assert body["candidate_languages"] == ["en", "hi", "or"]
    assert body["lid_model"] == "mock-lid"
    assert body["code_mixed_segments"] == 0


async def test_language_hint_restricts_candidates(client: AsyncClient) -> None:
    meeting_id = await _process(client, 12, languages="en,or")

    body = (await client.get(f"{URL}/{meeting_id}/languages")).json()

    assert body["candidate_languages"] == ["en", "or"]
    assert {x["language"] for x in body["languages"]} <= {"en", "or"}


async def test_languages_before_processing_is_409_and_unknown_404(client: AsyncClient) -> None:
    upload = await client.post(URL, files={"file": ("m.wav", make_wav(seconds=3))})

    early = await client.get(f"{URL}/{upload.json()['meeting_id']}/languages")

    assert early.status_code == 409
    assert early.json()["error"]["code"] == "language_summary_not_available"
    assert (await client.get(f"{URL}/{uuid4()}/languages")).status_code == 404
