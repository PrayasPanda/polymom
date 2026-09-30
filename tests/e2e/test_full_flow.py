"""Happy path through the real API and workers: upload -> process -> every read endpoint."""

import json

import httpx
import jsonschema

from tests.conftest import make_wav
from tests.e2e.conftest import upload, wait_for


def _processed_meeting(api: httpx.Client) -> str:
    created = upload(
        api, make_wav(seconds=20), title="E2E budget review", expected_speakers="2",
        languages="en,hi,or",
    )
    assert created.status_code == 202, created.text
    meeting_id: str = created.json()["meeting_id"]
    assert api.post(f"/meetings/{meeting_id}/process").status_code == 202
    assert wait_for(api, meeting_id)["status"] == "completed"
    return meeting_id


def test_full_flow(api: httpx.Client) -> None:
    meeting_id = _processed_meeting(api)
    base = f"/meetings/{meeting_id}"

    status = api.get(f"{base}/status").json()
    assert status["status"] == "completed"
    assert status["percent"] == 100.0

    # SSE: a finished meeting still emits its final snapshot, then the stream ends.
    with api.stream("GET", f"{base}/status/stream", timeout=30) as stream:
        assert stream.headers["content-type"].startswith("text/event-stream")
        data = next(line for line in stream.iter_lines() if line.startswith("data:"))
    assert json.loads(data[5:])["meeting_id"] == meeting_id

    renamed = api.patch(f"{base}/speakers", json={"names": {"Person 1": "Ravi"}})
    assert renamed.status_code == 200

    transcript = api.get(f"{base}/transcript").json()
    assert transcript["utterances"]
    assert {"en", "hi", "or"} <= {u["primary_language"] for u in transcript["utterances"]}
    for fmt, marker in (("txt", "Ravi"), ("srt", "-->"), ("vtt", "WEBVTT"), ("md", "#")):
        body = api.get(f"{base}/transcript", params={"format": fmt})
        assert body.status_code == 200, fmt
        assert marker in body.text, fmt

    analytics = api.get(f"{base}/analytics").json()
    assert analytics["meeting_stats"]["num_speakers"] == 2
    assert "Person 1" in api.get(f"{base}/analytics", params={"format": "csv"}).text

    summary = api.get(f"{base}/summary").json()
    assert summary["executive_summary"]
    assert summary["verification_report"]["evidence_checked"] >= 1
    assert api.get(f"{base}/summary", params={"format": "md"}).status_code == 200

    languages = api.get(f"{base}/languages").json()
    assert {s["language"] for s in languages["languages"]} >= {"en", "hi"}

    for fmt, magic in (("pdf", b"%PDF"), ("docx", b"PK"), ("md", b"#"), ("json", b"{")):
        export = api.get(f"{base}/export", params={"format": fmt})
        assert export.status_code == 200, fmt
        assert export.content.startswith(magic), fmt

    assert api.get(f"{base}/audio").content[:4] == b"RIFF"

    hits = api.get("/search", params={"q": "budget", "meeting_id": meeting_id}).json()
    assert hits["items"]

    assert api.delete(base).status_code == 204
    assert api.get(base).status_code == 404


def test_result_matches_published_json_schema(api: httpx.Client) -> None:
    """Contract: /result validates against the schema served at /schema/meeting-result."""
    meeting_id = _processed_meeting(api)
    schema = api.get("/schema/meeting-result").json()
    result = api.get(f"/meetings/{meeting_id}/result").json()
    jsonschema.validate(result, schema)
    assert result["transcript"] and result["analytics"] and result["summary"]
    api.delete(f"/meetings/{meeting_id}")


def test_ui_is_served(api: httpx.Client) -> None:
    root = str(api.base_url).removesuffix("/api/v1/")
    page = httpx.get(root + "/", timeout=10)
    assert page.status_code == 200
    assert "Upload a meeting" in page.text
    assert (
        httpx.get(root + "/ui/partials/meetings", headers=dict(api.headers)).status_code == 200
    )
