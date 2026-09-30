"""End-to-end fixtures: a running docker compose stack (API, workers, Postgres, Redis).

Start it with ``make e2e-up`` (mock ML backends, see docker/docker-compose.e2e.yml),
then ``make e2e``. Every test here is skipped when the API is not reachable, so the
default ``pytest`` run is unaffected.
"""

import os
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest

ROOT = Path(__file__).parents[2]
BASE_URL = os.environ.get("E2E_BASE_URL", "http://localhost:8000")
COMPOSE = [
    "docker",
    "compose",
    "-f",
    str(ROOT / "docker" / "docker-compose.yml"),
    "-f",
    str(ROOT / "docker" / "docker-compose.e2e.yml"),
]
TERMINAL = {"completed", "completed_with_errors", "failed", "cancelled"}


def compose(*args: str, env: dict[str, str] | None = None, timeout: float = 300) -> str:
    """Run ``docker compose`` against the e2e stack and return stdout."""
    result = subprocess.run(  # noqa: S603
        [*COMPOSE, *args],
        capture_output=True,
        text=True,
        check=True,
        timeout=timeout,
        env={**os.environ, **(env or {})},
    )
    return result.stdout


def _reachable() -> bool:
    try:
        return httpx.get(f"{BASE_URL}/api/v1/health/ready", timeout=3).status_code == 200
    except httpx.HTTPError:
        return False


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    here = Path(__file__).parent
    for item in items:
        if here in Path(str(item.fspath)).parents:
            item.add_marker(pytest.mark.e2e)


@pytest.fixture(scope="session")
def api_key() -> str:
    if not _reachable():
        pytest.skip(f"e2e stack not running at {BASE_URL} (make e2e-up)")
    if key := os.environ.get("E2E_API_KEY"):
        return key
    out = compose("exec", "-T", "api", "python", "-m", "scripts.create_api_key", "--label", "e2e")
    return out.strip().splitlines()[-1].strip()


@pytest.fixture
def api(api_key: str) -> Iterator[httpx.Client]:
    with httpx.Client(
        base_url=f"{BASE_URL}/api/v1", headers={"X-API-Key": api_key}, timeout=60
    ) as client:
        yield client


def upload(api: httpx.Client, data: bytes, name: str = "m.wav", **form: Any) -> httpx.Response:
    return api.post(
        "/meetings",
        files={"file": (name, data)},
        data=form,
        params={"allow_duplicate": "true"},
    )


def wait_for(
    api: httpx.Client, meeting_id: str, statuses: set[str] = TERMINAL, timeout: float = 180
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body: dict[str, Any] = api.get(f"/meetings/{meeting_id}").json()
        if body["status"] in statuses:
            return body
        time.sleep(0.5)
    raise AssertionError(f"meeting {meeting_id} did not reach {statuses} in {timeout}s")


def wait_for_stage(api: httpx.Client, meeting_id: str, stage: str, timeout: float = 120) -> None:
    """Block until ``stage`` of the default run has completed."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        runs = api.get(f"/meetings/{meeting_id}/runs").json()["items"]
        if runs:
            result = api.get(
                f"/meetings/{meeting_id}/runs/{runs[0]['run_id']}", params={"include": ""}
            )
            stages = (result.json().get("processing") or {}).get("stages", [])
            if any(s["name"] == stage and s["status"] == "completed" for s in stages):
                return
        time.sleep(0.5)
    raise AssertionError(f"stage {stage} did not complete in {timeout}s")
