"""Failure flows against the running stack.

Upload validation failures need nothing special. Cancel, resume and LLM failure are
driven at the infrastructure level: a worker is stopped, restarted, or started with
an unreachable LLM, exactly as it would happen in production. Those tests restore
the stack before they finish.
"""

import struct
import subprocess
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest

from tests.conftest import make_wav
from tests.e2e.conftest import compose, upload, wait_for, wait_for_stage


def _error_code(response: httpx.Response) -> str:
    code: str = response.json()["error"]["code"]
    return code


def _ffmpeg(tmp_path: Path, name: str, *args: str) -> bytes:
    out = tmp_path / name
    subprocess.run(  # noqa: S603
        ["ffmpeg", "-v", "error", "-y", *args, str(out)],  # noqa: S607
        check=True,
        timeout=60,
    )
    return out.read_bytes()


def test_corrupted_file_is_rejected(api: httpx.Client) -> None:
    garbage = b"RIFF" + struct.pack("<I", 4096) + b"WAVE" + b"\xde\xad\xbe\xef" * 1024
    response = upload(api, garbage)
    assert response.status_code == 422
    assert _error_code(response) == "corrupted_media"


def test_unsupported_type_is_rejected(api: httpx.Client) -> None:
    response = upload(api, b"hello, not audio", name="notes.txt")
    assert response.status_code == 415
    assert _error_code(response) == "unsupported_file_type"


def test_video_without_audio_is_rejected(api: httpx.Client, tmp_path: Path) -> None:
    video = _ffmpeg(tmp_path, "silent.mp4", "-f", "lavfi", "-i", "color=c=black:s=32x32:d=1", "-an")
    response = upload(api, video, name="silent.mp4")
    assert response.status_code == 422
    assert _error_code(response) == "corrupted_media"


def test_over_limit_duration_is_rejected(api: httpx.Client) -> None:
    """The e2e overlay sets MAX_AUDIO_DURATION_MINUTES=1."""
    response = upload(api, make_wav(seconds=65, rate=8000))
    assert response.status_code == 422
    assert _error_code(response) == "audio_too_long"


def test_missing_api_key_is_rejected(api: httpx.Client) -> None:
    response = httpx.get(str(api.base_url) + "meetings", timeout=10)
    assert response.status_code == 401


@pytest.fixture
def gpu_worker_stopped() -> Iterator[None]:
    """Stop the GPU worker so a run parks after preprocessing; always restart it."""
    compose("stop", "worker-gpu")
    try:
        yield
    finally:
        compose("start", "worker-gpu")


@pytest.mark.slow
def test_cancel_mid_run(api: httpx.Client, gpu_worker_stopped: None) -> None:
    meeting_id = upload(api, make_wav(seconds=10)).json()["meeting_id"]
    api.post(f"/meetings/{meeting_id}/process")
    wait_for_stage(api, meeting_id, "preprocess")  # the run is now waiting for diarization

    assert api.post(f"/meetings/{meeting_id}/cancel").status_code == 202
    compose("start", "worker-gpu")

    assert wait_for(api, meeting_id)["status"] == "cancelled"
    api.delete(f"/meetings/{meeting_id}")


@pytest.mark.slow
def test_resume_after_worker_restart(api: httpx.Client, gpu_worker_stopped: None) -> None:
    meeting_id = upload(api, make_wav(seconds=10)).json()["meeting_id"]
    api.post(f"/meetings/{meeting_id}/process")
    wait_for_stage(api, meeting_id, "preprocess")
    compose("restart", "worker-cpu")  # a restart must not lose or redo finished stages
    compose("start", "worker-gpu")

    assert wait_for(api, meeting_id)["status"] == "completed"
    result = api.get(f"/meetings/{meeting_id}/result").json()
    preprocess_runs = [s for s in result["processing"]["stages"] if s["name"] == "preprocess"]
    assert len(preprocess_runs) == 1
    assert preprocess_runs[0]["status"] == "completed"
    api.delete(f"/meetings/{meeting_id}")


@pytest.fixture
def broken_llm() -> Iterator[None]:
    """Run worker-llm against an unreachable OpenAI-compatible endpoint."""
    env = {
        "LLM_PROVIDER": "openai",
        "LLM_API_KEY": "sk-e2e-not-a-real-key",
        "OPENAI_BASE_URL": "http://127.0.0.1:9/v1",
        "LLM_MAX_RETRIES": "0",
        "MAX_RETRIES": "0",
    }
    compose("up", "-d", "--no-deps", "--wait", "worker-llm", env=env)
    try:
        yield
    finally:
        compose("up", "-d", "--no-deps", "--wait", "worker-llm")


@pytest.mark.slow
def test_llm_failure_completes_with_errors(api: httpx.Client, broken_llm: None) -> None:
    meeting_id = upload(api, make_wav(seconds=12)).json()["meeting_id"]
    api.post(f"/meetings/{meeting_id}/process")

    meeting = wait_for(api, meeting_id)
    assert meeting["status"] == "completed_with_errors"
    # Everything before the summary is still served.
    assert api.get(f"/meetings/{meeting_id}/transcript").status_code == 200
    assert api.get(f"/meetings/{meeting_id}/analytics").status_code == 200
    summary = api.get(f"/meetings/{meeting_id}/summary")
    assert summary.status_code == 409
    api.delete(f"/meetings/{meeting_id}")
