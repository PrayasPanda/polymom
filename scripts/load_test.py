"""Locust load test for the upload and status endpoints (run against the mock backends).

    # stack with mock backends, then:
    POLYMOM_API_KEY=pk_... uv run locust -f scripts/load_test.py --host http://localhost:8000 \\
        --headless -u 50 -r 10 -t 2m --csv docs/load/results

Uploads are small generated WAVs, so the test measures the API, validation,
storage and enqueue path, not ML throughput (that scales with workers, not the API).
"""

import io
import math
import os
import random
import struct
import wave

from locust import HttpUser, between, task

API = "/api/v1/meetings"


def tiny_wav(seconds: float = 2.0, rate: int = 16000) -> bytes:
    """A unique short tone so every upload has a different SHA-256 (no dedupe hits)."""
    frequency = random.uniform(200, 800)  # noqa: S311 - not crypto
    frames = int(seconds * rate)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(
            b"".join(
                struct.pack("<h", int(8000 * math.sin(2 * math.pi * frequency * i / rate)))
                for i in range(frames)
            )
        )
    return buf.getvalue()


class MeetingUser(HttpUser):
    wait_time = between(0.5, 2.0)

    def on_start(self) -> None:
        key = os.environ.get("POLYMOM_API_KEY")
        if key:
            self.client.headers["X-API-Key"] = key
        self.meetings: list[str] = []

    @task(1)
    def upload_and_process(self) -> None:
        with self.client.post(
            API,
            files={"file": ("load.wav", tiny_wav(), "audio/wav")},
            name="POST /meetings",
            catch_response=True,
        ) as response:
            if response.status_code == 429:
                response.success()  # rate limiting working as configured is not a failure
                return
            if response.status_code not in (200, 202):
                response.failure(f"upload {response.status_code}")
                return
            meeting_id = response.json()["meeting_id"]
        self.meetings.append(meeting_id)
        self.client.post(f"{API}/{meeting_id}/process", name="POST /meetings/{id}/process")

    @task(8)
    def poll_status(self) -> None:
        if not self.meetings:
            return
        meeting_id = random.choice(self.meetings)  # noqa: S311
        self.client.get(f"{API}/{meeting_id}/status", name="GET /meetings/{id}/status")

    @task(2)
    def list_meetings(self) -> None:
        self.client.get(API, params={"limit": 20}, name="GET /meetings")
