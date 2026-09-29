import csv
import io
import sys
from uuid import uuid4

import pytest
from httpx import AsyncClient

from tests.conftest import make_wav, requires_ffmpeg

pytestmark = requires_ffmpeg

URL = "/api/v1/meetings"


async def _process(client: AsyncClient, seconds: float = 18) -> str:
    upload = await client.post(URL, files={"file": ("m.wav", make_wav(seconds=seconds))})
    meeting_id: str = upload.json()["meeting_id"]
    await client.post(f"{URL}/{meeting_id}/process")
    assert (await client.get(f"{URL}/{meeting_id}")).json()["status"] == "completed"
    return meeting_id


async def test_analytics_json_and_csv(client: AsyncClient) -> None:
    # Mock diarization alternates Person 1 / Person 2 every 2 s with no overlap.
    meeting_id = await _process(client)
    await client.patch(f"{URL}/{meeting_id}/speakers", json={"names": {"Person 1": "Ravi"}})

    body = (await client.get(f"{URL}/{meeting_id}/analytics")).json()

    assert body["meeting_id"] == meeting_id
    assert body["speaker_names"] == {"Person 1": "Ravi"}
    m = body["meeting_stats"]
    assert m["meeting_duration_seconds"] == 18.0
    assert m["num_speakers"] == 2
    assert m["overlap_seconds"] == 0.0
    assert m["total_turn_switches"] == 8  # 9 two-second turns
    assert m["participation_balance"] == "balanced"
    speakers = body["speakers"]
    assert [s["speaker"] for s in speakers] == ["Person 1", "Person 2"]  # 10 s vs 8 s
    assert speakers[0]["speaker_name"] == "Ravi"
    assert speakers[1]["speaker_name"] is None
    assert sum(s["speaking_time_seconds"] for s in speakers) == m["total_speech_seconds"]
    assert sum(s["segment_share_percent"] for s in speakers) == pytest.approx(100, abs=0.01)
    assert sum(s["word_count"] for s in speakers) == m["total_words"] > 0
    assert body["timeline"][0]["speakers"] == {"Person 1": 10.0, "Person 2": 8.0}
    assert "speaking_time_seconds" in body["metric_definitions"]

    response = await client.get(f"{URL}/{meeting_id}/analytics", params={"format": "csv"})
    assert response.headers["content-type"].startswith("text/csv")
    rows = list(csv.DictReader(io.StringIO(response.text)))
    assert [(r["speaker"], r["display_name"]) for r in rows] == [
        ("Person 1", "Ravi"),
        ("Person 2", "Person 2"),
    ]
    assert rows[0]["speaking_time_seconds"] == "10.0"


async def test_analytics_errors(client: AsyncClient) -> None:
    upload = await client.post(URL, files={"file": ("m.wav", make_wav(seconds=3))})
    meeting_id = upload.json()["meeting_id"]

    early = await client.get(f"{URL}/{meeting_id}/analytics")
    assert early.status_code == 409
    assert early.json()["error"]["code"] == "analytics_not_available"
    assert (await client.get(f"{URL}/{uuid4()}/analytics")).status_code == 404
    bad_chart = await client.get(f"{URL}/{meeting_id}/analytics/charts/pie3d")
    assert bad_chart.status_code == 422


@pytest.mark.parametrize("chart", ["speaking-time", "timeline"])
async def test_chart_endpoint(client: AsyncClient, chart: str) -> None:
    pytest.importorskip("matplotlib")
    meeting_id = await _process(client, 8)

    response = await client.get(f"{URL}/{meeting_id}/analytics/charts/{chart}")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content.startswith(b"\x89PNG")


async def test_chart_endpoint_without_viz_extra(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    meeting_id = await _process(client, 4)
    monkeypatch.setitem(sys.modules, "matplotlib.figure", None)

    response = await client.get(f"{URL}/{meeting_id}/analytics/charts/speaking-time")

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "charts_unavailable"
