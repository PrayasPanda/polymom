from uuid import uuid4

from httpx import AsyncClient, Response

from tests.conftest import make_wav, requires_ffmpeg

pytestmark = requires_ffmpeg

URL = "/api/v1/meetings"


async def _process(client: AsyncClient, seconds: float = 18, **form: str) -> str:
    upload = await client.post(URL, files={"file": ("m.wav", make_wav(seconds=seconds))}, data=form)
    meeting_id: str = upload.json()["meeting_id"]
    await client.post(f"{URL}/{meeting_id}/process")
    assert (await client.get(f"{URL}/{meeting_id}")).json()["status"] == "completed"
    return meeting_id


async def test_speaker_transcript_json(client: AsyncClient) -> None:
    # Mock diarization alternates Person 1 / Person 2 every 2 s; mock ASR emits 3 s segments.
    meeting_id = await _process(client)

    body = (await client.get(f"{URL}/{meeting_id}/transcript")).json()

    assert body["meeting_id"] == meeting_id
    assert body["speakers"] == ["Person 1", "Person 2"]
    utterances = body["utterances"]
    assert utterances
    assert [(u["start"], u["speaker"]) for u in utterances] == sorted(
        (u["start"], u["speaker"]) for u in utterances
    )
    assert {u["speaker"] for u in utterances} == {"Person 1", "Person 2"}
    assert all(w["speaker"] == u["speaker"] for u in utterances for w in u["words"])
    assert {u["primary_language"] for u in utterances} == {"en", "hi", "or"}
    stats = body["alignment_stats"]
    assert stats["percent_assigned"] == 100.0
    assert stats["percent_unknown"] == 0.0
    assert body["total_duration"] == 18.0
    assert body["speaker_names"] == {}
    raw_words = sum(
        len(s["words"])
        for s in (
            await client.get(f"{URL}/{meeting_id}/transcript", params={"view": "raw"})
        ).json()["segments"]
    )
    assert stats["total_words"] == raw_words  # every ASR word is attributed exactly once


async def test_all_formats_and_speaker_renaming(client: AsyncClient) -> None:
    meeting_id = await _process(client, 12)

    async def get(fmt: str) -> Response:
        return await client.get(f"{URL}/{meeting_id}/transcript", params={"format": fmt})

    rename = await client.patch(
        f"{URL}/{meeting_id}/speakers", json={"names": {"Person 1": " Ravi "}}
    )
    assert rename.status_code == 200
    assert rename.json() == {"meeting_id": meeting_id, "speaker_names": {"Person 1": "Ravi"}}

    body = (await get("json")).json()
    assert body["speaker_names"] == {"Person 1": "Ravi"}
    assert {(u["speaker"], u["speaker_name"]) for u in body["utterances"]} == {
        ("Person 1", "Ravi"),
        ("Person 2", None),
    }

    txt = await get("txt")
    assert txt.headers["content-type"].startswith("text/plain")
    assert txt.text.startswith("[00:00:00 - ")
    assert "] Ravi: " in txt.text
    assert "] Person 2: " in txt.text

    srt = await get("srt")
    assert srt.headers["content-type"].startswith("application/x-subrip")
    assert srt.text.startswith("1\n00:00:00,000 --> ")
    assert "Ravi: " in srt.text

    vtt = await get("vtt")
    assert vtt.headers["content-type"].startswith("text/vtt")
    assert vtt.headers["content-disposition"] == f'attachment; filename="{meeting_id}.vtt"'
    assert vtt.text.startswith("WEBVTT\n\n00:00:00.000 --> ")
    assert "<v Ravi>" in vtt.text
    assert "<v Person 2>" in vtt.text

    md = await get("md")
    assert md.headers["content-type"].startswith("text/markdown")
    assert md.text.startswith("# Transcript\n\n**Ravi** · 00:00:00")

    cleared = await client.patch(f"{URL}/{meeting_id}/speakers", json={"names": {"Person 1": None}})
    assert cleared.json()["speaker_names"] == {}
    assert "Person 1: " in (await get("txt")).text


async def test_rename_validation(client: AsyncClient) -> None:
    meeting_id = await _process(client, 6)

    bad = await client.patch(f"{URL}/{meeting_id}/speakers", json={"names": {"Person 9": "X"}})

    assert bad.status_code == 422
    assert bad.json()["error"]["details"] == {
        "unknown": ["Person 9"],
        "speakers": ["Person 1", "Person 2"],
    }
    assert (await client.patch(f"{URL}/{uuid4()}/speakers", json={"names": {}})).status_code == 404


async def test_rename_before_processing_is_409(client: AsyncClient) -> None:
    upload = await client.post(URL, files={"file": ("m.wav", make_wav(seconds=3))})

    response = await client.patch(
        f"{URL}/{upload.json()['meeting_id']}/speakers", json={"names": {"Person 1": "A"}}
    )

    assert response.status_code == 409


async def test_speaker_view_before_processing_and_raw_only_formats(client: AsyncClient) -> None:
    upload = await client.post(URL, files={"file": ("m.wav", make_wav(seconds=3))})
    meeting_id = upload.json()["meeting_id"]

    early = await client.get(f"{URL}/{meeting_id}/transcript")
    assert early.status_code == 409
    assert "view=raw" in early.json()["error"]["message"]

    processed = await _process(client, 6)
    raw_vtt = await client.get(
        f"{URL}/{processed}/transcript", params={"view": "raw", "format": "vtt"}
    )
    assert raw_vtt.status_code == 422
