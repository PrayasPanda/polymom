from httpx import AsyncClient

from app import __version__


async def test_health_returns_app_metadata(client: AsyncClient) -> None:
    response = await client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "app_name": "polymom",
        "version": __version__,
        "env": "test",
    }


async def test_unknown_meeting_returns_error_envelope(client: AsyncClient) -> None:
    response = await client.get("/api/v1/meetings/00000000-0000-4000-8000-000000000000")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "meeting_not_found"
