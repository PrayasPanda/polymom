"""Seed a demo: create an API key, upload a sample meeting, process it, print the UI URL.

Runs inside the API container (``make demo``), so the host only needs Docker::

    docker compose -f docker/docker-compose.yml exec api python -m scripts.demo
    python -m scripts.demo --audio data/eval/meetings/codemixed_01.wav   # your own file

Without ``--audio`` it synthesises a 60 s two-voice WAV; with the default mock
backends the transcript is the scripted English/Hindi/Odia demo dialogue.
"""

import argparse
import asyncio
import io
import math
import struct
import sys
import time
import wave
from pathlib import Path

import httpx

from app.core.config import Settings
from app.db.session import create_engine, create_sessionmaker
from app.repositories.unit_of_work import unit_of_work_factory

TERMINAL = {"completed", "completed_with_errors", "failed", "cancelled"}


def synth_meeting(seconds: int = 60, rate: int = 16000) -> bytes:
    """Alternating 3 s 'voices' (two pitches with a syllable-like envelope) and short pauses."""
    frames = bytearray()
    for i in range(seconds * rate):
        t = i / rate
        slot = int(t // 3.5)
        if t % 3.5 > 3.0:  # pause between turns
            frames += struct.pack("<h", 0)
            continue
        pitch = 140 if slot % 2 == 0 else 230
        envelope = 0.5 + 0.5 * math.sin(2 * math.pi * 4 * t)
        frames += struct.pack("<h", int(6000 * envelope * math.sin(2 * math.pi * pitch * t)))
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(bytes(frames))
    return buf.getvalue()


async def create_key(label: str) -> str:
    settings = Settings()
    engine = create_engine(settings.resolved_database_url)
    try:
        async with unit_of_work_factory(create_sessionmaker(engine))() as uow:
            _, token = await uow.api_keys.create(label=label, owner="demo")
            await uow.commit()
        return token
    finally:
        await engine.dispose()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--api", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--public-url", default="http://localhost:8000", help="URL to print")
    parser.add_argument("--audio", type=Path, help="Recording to upload (default: synthetic)")
    parser.add_argument("--timeout", type=float, default=900)
    args = parser.parse_args(argv)

    key = asyncio.run(create_key("demo"))
    data = args.audio.read_bytes() if args.audio else synth_meeting()
    name = args.audio.name if args.audio else "demo-meeting.wav"
    with httpx.Client(base_url=f"{args.api}/api/v1", headers={"X-API-Key": key}) as api:
        created = api.post(
            "/meetings",
            files={"file": (name, data)},
            data={"title": "Demo: Q3 budget review", "languages": "en,hi,or"},
            params={"allow_duplicate": "true"},
            timeout=120,
        )
        created.raise_for_status()
        meeting_id = created.json()["meeting_id"]
        api.post(f"/meetings/{meeting_id}/process").raise_for_status()
        deadline = time.monotonic() + args.timeout
        status = "queued"
        while status not in TERMINAL and time.monotonic() < deadline:
            time.sleep(1)
            status = api.get(f"/meetings/{meeting_id}").json()["status"]
            print(f"\r  status: {status:<24}", end="", flush=True)
        print()
        api.patch(
            f"/meetings/{meeting_id}/speakers",
            json={"names": {"Person 1": "Ravi", "Person 2": "Sunita"}},
        )

    print(
        f"""
Polymom demo is ready ({status}).

  UI:       {args.public_url}/
  Meeting:  {args.public_url}/meetings/{meeting_id}
  API docs: {args.public_url}/docs
  API key:  {key}

Paste the API key into the box at the top of the UI.
"""
    )
    return 0 if status in {"completed", "completed_with_errors"} else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
