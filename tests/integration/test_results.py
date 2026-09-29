"""Consolidated result, run history, filters, search, dedupe, exports and cascade delete."""

import io
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from httpx import AsyncClient
from pypdf import PdfReader

from app.core.config import Settings
from app.schemas.result import MeetingResult
from tests.conftest import make_wav, requires_ffmpeg

pytestmark = requires_ffmpeg

URL = "/api/v1/meetings"


async def _upload(client: AsyncClient, seconds: float = 12, **form: str) -> str:
    response = await client.post(
        URL,
        files={"file": ("m.wav", make_wav(seconds=seconds))},
        data=form,
        params={"allow_duplicate": "true"},
    )
    meeting_id: str = response.json()["meeting_id"]
    return meeting_id


async def _process(client: AsyncClient, seconds: float = 12, **form: str) -> str:
    meeting_id = await _upload(client, seconds, **form)
    await client.post(f"{URL}/{meeting_id}/process")
    return meeting_id


async def test_consolidated_result_and_include(client: AsyncClient) -> None:
    meeting_id = await _process(client, 18, title="Budget review")
    await client.patch(f"{URL}/{meeting_id}/speakers", json={"names": {"Person 1": "Ravi"}})

    body = (await client.get(f"{URL}/{meeting_id}/result")).json()

    result = MeetingResult.model_validate(body)  # round-trips through the published model
    assert result.schema_version == "1.0.0"
    assert result.meeting.title == "Budget review"
    assert result.meeting.detected_languages == ["en", "hi", "or"]
    assert result.meeting.num_speakers == 2
    assert result.processing is not None
    assert result.processing.status == "completed"
    assert {s.name for s in result.processing.stages} >= {
        "preprocess",
        "diarize",
        "align",
        "summarize",
    }
    assert result.processing.model_versions["diarize"] == "mock"
    assert set(result.processing.timings_ms) >= {"preprocess", "align"}
    assert [(s.label, s.display_name) for s in result.speakers] == [
        ("Person 1", "Ravi"),
        ("Person 2", None),
    ]
    assert result.speakers[0].stats is not None
    assert result.transcript is not None
    assert result.transcript.utterances[0].speaker_name in {"Ravi", None}
    assert result.analytics is not None
    assert result.summary is not None
    assert result.verification_report == result.summary.verification_report
    assert result.languages is not None
    assert result.included == ["transcript", "analytics", "summary"]

    trimmed = (await client.get(f"{URL}/{meeting_id}/result", params={"include": "summary"})).json()
    assert (trimmed["transcript"], trimmed["analytics"]) == (None, None)
    assert trimmed["summary"] is not None
    assert trimmed["included"] == ["summary"]
    bad = await client.get(f"{URL}/{meeting_id}/result", params={"include": "audio"})
    assert bad.status_code == 422

    schema = (await client.get("/api/v1/schema/meeting-result")).json()
    assert schema["version"] == "1.0.0"
    assert schema["title"] == "MeetingResult"
    assert {"meeting", "processing", "speakers", "summary", "schema_version"} <= set(
        schema["properties"]
    )


async def test_result_before_processing(client: AsyncClient) -> None:
    meeting_id = await _upload(client, 3)
    body = (await client.get(f"{URL}/{meeting_id}/result")).json()
    assert body["processing"] is None
    assert body["summary"] is None
    assert (await client.get(f"{URL}/{meeting_id}/utterances")).status_code == 409
    assert (await client.get(f"{URL}/{uuid4()}/result")).status_code == 404


async def test_run_history(client: AsyncClient) -> None:
    meeting_id = await _process(client, 6)
    first = (await client.get(f"{URL}/{meeting_id}/runs")).json()["items"][0]["run_id"]
    await client.post(f"{URL}/{meeting_id}/process", params={"force": "true"})

    runs = (await client.get(f"{URL}/{meeting_id}/runs")).json()["items"]

    assert len(runs) == 2
    assert runs[1]["run_id"] == first
    assert [r["is_default"] for r in runs] == [True, False]
    old = (await client.get(f"{URL}/{meeting_id}/runs/{first}")).json()
    assert old["processing"]["run_id"] == first
    pinned = (await client.get(f"{URL}/{meeting_id}/result", params={"run_id": first})).json()
    assert pinned["processing"]["run_id"] == first
    missing = await client.get(f"{URL}/{meeting_id}/runs/{uuid4()}")
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "run_not_found"
    utterances = await client.get(f"{URL}/{meeting_id}/utterances", params={"run_id": first})
    assert utterances.json()["run_id"] == first


async def test_utterance_filters(client: AsyncClient) -> None:
    meeting_id = await _process(client, 12)
    get = lambda **p: client.get(f"{URL}/{meeting_id}/utterances", params=p)  # noqa: E731

    everything = (await get()).json()
    assert everything["total"] == len(everything["items"]) > 2
    by_speaker = (await get(speaker="Person 2")).json()
    assert {u["speaker"] for u in by_speaker["items"]} == {"Person 2"}
    hindi = (await get(language="hi")).json()
    assert hindi["items"]
    assert {u["primary_language"] for u in hindi["items"]} == {"hi"}
    window = (await get(start=3, end=6)).json()
    assert all(u["end"] >= 3 and u["start"] <= 6 for u in window["items"])
    assert window["total"] < everything["total"]
    text = (await get(q="MORNING")).json()
    assert text["items"]
    assert all("morning" in u["text"].lower() for u in text["items"])
    page = (await get(limit=1, offset=1)).json()
    assert page["items"][0] == everything["items"][1]


async def test_list_filters_sort_and_cursor(client: AsyncClient) -> None:
    processed = await _process(client, 18)
    short = await _upload(client, 2)
    long = await _upload(client, 30)

    async def ids(**params: str) -> list[str]:
        return [m["meeting_id"] for m in (await client.get(URL, params=params)).json()["items"]]

    assert await ids(status="completed") == [processed]
    assert await ids(status="queued") == [long, short]
    assert await ids(language="hi") == [processed]
    assert await ids(language="or", min_speakers="2", max_speakers="2") == [processed]
    assert await ids(min_speakers="3") == []
    assert await ids(sort="duration_seconds", order="asc") == [short, processed, long]
    future = (datetime.now(UTC) + timedelta(days=1)).isoformat()
    past = (datetime.now(UTC) - timedelta(days=1)).isoformat()
    assert len(await ids(created_from=past, created_to=future)) == 3
    assert await ids(created_from=future) == []

    seen: list[str] = []
    cursor = None
    while True:
        params = {"limit": "1", "sort": "duration_seconds"} | ({"cursor": cursor} if cursor else {})
        page = (await client.get(URL, params=params)).json()
        seen += [m["meeting_id"] for m in page["items"]]
        cursor = page["next_cursor"]
        if cursor is None:
            break
    assert seen == [long, processed, short]
    bad = await client.get(URL, params={"cursor": "not-a-cursor"})
    assert bad.status_code == 422


@pytest.mark.parametrize(
    ("query", "script"),
    [("morning", "Latin"), ("नमस्ते", "Devanagari"), ("ନମସ୍କାର", "Odia"), ("को", "short")],
)
async def test_search_in_three_scripts(client: AsyncClient, query: str, script: str) -> None:
    meeting_id = await _process(client, 18, title="Standup")
    other = await _process(client, 3)

    body = (await client.get("/api/v1/search", params={"q": query})).json()

    hits = [h for h in body["items"] if h["kind"] == "utterance"]
    assert hits, script
    assert all(query.lower() in h["text"].lower() for h in hits)
    assert {h["meeting_id"] for h in hits} <= {meeting_id, other}
    assert hits[0]["meeting_title"] in {"Standup", None}
    scoped = (
        await client.get("/api/v1/search", params={"q": query, "meeting_id": meeting_id})
    ).json()
    assert {h["meeting_id"] for h in scoped["items"]} == {meeting_id}


async def test_search_finds_summaries(client: AsyncClient) -> None:
    await _process(client, 12)
    body = (await client.get("/api/v1/search", params={"q": "mock summary"})).json()
    assert [h["kind"] for h in body["items"]] == ["summary"]


async def test_duplicate_upload_is_idempotent(client: AsyncClient) -> None:
    data = make_wav(seconds=1)
    first = await client.post(URL, files={"file": ("a.wav", data)})
    again = await client.post(URL, files={"file": ("renamed.wav", data)})
    forced = await client.post(
        URL, files={"file": ("a.wav", data)}, params={"allow_duplicate": "true"}
    )

    assert (first.status_code, first.json()["duplicate"]) == (202, False)
    assert (again.status_code, again.json()["duplicate"]) == (200, True)
    assert again.json()["meeting_id"] == first.json()["meeting_id"]
    assert forced.status_code == 202
    assert forced.json()["meeting_id"] != first.json()["meeting_id"]
    assert (await client.get(URL)).json()["total"] == 2
    meta = (await client.get(f"{URL}/{first.json()['meeting_id']}")).json()
    assert len(meta["sha256"]) == 64


async def test_delete_cascades_rows_artifacts_and_search(
    client: AsyncClient, settings: Settings
) -> None:
    meeting_id = await _process(client, 6)
    await client.get(f"{URL}/{meeting_id}/export", params={"format": "md"})
    folder = settings.storage_dir / "meetings" / meeting_id
    assert any(folder.rglob("*.md"))

    assert (await client.delete(f"{URL}/{meeting_id}")).status_code == 204

    assert not folder.exists()
    assert (await client.get(f"{URL}/{meeting_id}/runs")).status_code == 404
    hits = (await client.get("/api/v1/search", params={"q": "morning"})).json()["items"]
    assert all(h["meeting_id"] != meeting_id for h in hits)


@pytest.mark.parametrize(
    ("fmt", "media_type", "magic"),
    [
        ("pdf", "application/pdf", b"%PDF"),
        ("docx", "application/vnd.openxmlformats", b"PK"),
        ("md", "text/markdown", b"# "),
        ("json", "application/json", b"{"),
    ],
)
async def test_exports_are_cached_and_invalidated(
    client: AsyncClient, settings: Settings, fmt: str, media_type: str, magic: bytes
) -> None:
    meeting_id = await _process(client, 9)
    run_id = (await client.get(f"{URL}/{meeting_id}/runs")).json()["items"][0]["run_id"]
    cached = settings.storage_dir / "meetings" / meeting_id / run_id / "exports" / f"minutes.{fmt}"

    response = await client.get(f"{URL}/{meeting_id}/export", params={"format": fmt})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith(media_type)
    assert response.content.startswith(magic)
    assert f"minutes-{meeting_id}-{run_id}.{fmt}" in response.headers["content-disposition"]
    assert cached.read_bytes() == response.content
    again = await client.get(f"{URL}/{meeting_id}/export", params={"format": fmt})
    assert again.content == response.content

    await client.patch(f"{URL}/{meeting_id}/speakers", json={"names": {"Person 1": "Ravi"}})
    assert not cached.exists()  # display names changed
    renamed = await client.get(f"{URL}/{meeting_id}/export", params={"format": fmt})
    if fmt == "pdf":
        text = "".join(p.extract_text() for p in PdfReader(io.BytesIO(renamed.content)).pages)
        assert "Ravi" in text
    elif fmt in ("md", "json"):
        assert "Ravi" in renamed.text


async def test_charts_are_cached(client: AsyncClient, settings: Settings) -> None:
    pytest.importorskip("matplotlib")
    meeting_id = await _process(client, 6)
    run_id = (await client.get(f"{URL}/{meeting_id}/runs")).json()["items"][0]["run_id"]

    png = await client.get(f"{URL}/{meeting_id}/analytics/charts/timeline")

    cached = settings.storage_dir / "meetings" / meeting_id / run_id / "charts" / "timeline.png"
    assert cached.read_bytes() == png.content
    assert (
        await client.get(f"{URL}/{meeting_id}/analytics/charts/timeline")
    ).content == png.content
