"""Web UI pages, HTMX partials, static assets and the playback audio endpoint."""

from pathlib import Path

from httpx import AsyncClient

from app.core.config import Settings
from app.ui.routes import fmt_time
from tests.conftest import make_wav, requires_ffmpeg
from tests.integration.test_security import ClientFactory, create_key

URL = "/api/v1/meetings"


def test_fmt_time() -> None:
    assert fmt_time(None) == "--:--"
    assert fmt_time(75.4) == "01:15"
    assert fmt_time(3725) == "1:02:05"


async def test_pages_are_public_html_with_their_own_csp(client: AsyncClient) -> None:
    for path in ("/", "/meetings/00000000-0000-0000-0000-000000000000"):
        response = await client.get(path)
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/html")
        csp = response.headers["content-security-policy"]
        assert "script-src 'self'" in csp
        assert "unsafe-inline" not in csp
    assert 'name="languages" value="or"' in (await client.get("/")).text


async def test_static_assets_and_noto_fonts(client: AsyncClient) -> None:
    for path in ("/static/app.js", "/static/app.css", "/static/htmx.min.js"):
        assert (await client.get(path)).status_code == 200
    font = await client.get("/static/fonts/NotoSansOriya-Regular.ttf")
    assert font.status_code == 200
    assert len(font.content) > 10_000


async def test_partials_use_api_key_auth(tmp_path: Path, client_factory: ClientFactory) -> None:
    settings = Settings(
        _env_file=None,
        app_env="test",
        storage_dir=tmp_path / "s",
        api_key_required=True,
        pipeline_execution="inline",
    )
    async for client in client_factory(settings):
        assert (await client.get("/")).status_code == 200  # shell stays public
        denied = await client.get("/ui/partials/meetings")
        assert denied.status_code == 401
        token = await create_key(settings, "ui")
        ok = await client.get("/ui/partials/meetings", headers={"X-API-Key": token})
        assert ok.status_code == 200
        assert "No meetings yet" in ok.text


@requires_ffmpeg
async def test_meeting_list_detail_and_audio(client: AsyncClient) -> None:
    created = await client.post(
        URL, files={"file": ("m.wav", make_wav(seconds=18))}, data={"title": "UI sync"}
    )
    meeting_id = created.json()["meeting_id"]

    queued = await client.get(f"/ui/partials/meetings/{meeting_id}")
    assert 'hx-trigger="every 2s"' in queued.text  # progress view polls until done

    await client.post(f"{URL}/{meeting_id}/process")
    await client.patch(f"{URL}/{meeting_id}/speakers", json={"names": {"Person 1": "Ravi"}})

    listing = await client.get("/ui/partials/meetings")
    assert "UI sync" in listing.text
    assert "every 3s" not in listing.text  # nothing running, so no polling

    detail = (await client.get(f"/ui/partials/meetings/{meeting_id}")).text
    for expected in (
        "Minutes of meeting",
        "Action items",
        "Speaker analytics",
        'id="transcript"',
        "data-seek=",
        "Ravi",
        'lang="hi"',
        'data-format="docx"',
    ):
        assert expected in detail, expected
    assert "every 2s" not in detail

    audio = await client.get(f"{URL}/{meeting_id}/audio")
    assert audio.status_code == 200
    assert audio.headers["content-type"] == "audio/wav"
    assert audio.content[:4] == b"RIFF"


async def test_audio_unknown_meeting_is_404(client: AsyncClient) -> None:
    response = await client.get(f"{URL}/00000000-0000-0000-0000-000000000000/audio")
    assert response.status_code == 404
