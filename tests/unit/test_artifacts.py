"""Artifact stores: local files, and S3 via moto (in-memory AWS)."""

import uuid
from collections.abc import Iterator
from pathlib import Path

import boto3
import pytest
from moto import mock_aws

from app.repositories.artifacts import ArtifactStore, LocalArtifactStore, meeting_prefix, run_key
from app.repositories.artifacts.base import validate_key
from app.repositories.artifacts.s3 import S3ArtifactStore


@pytest.fixture
def local_store(tmp_path: Path) -> LocalArtifactStore:
    return LocalArtifactStore(tmp_path / "root")


@pytest.fixture
def s3_store() -> Iterator[S3ArtifactStore]:
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket="polymom")
        yield S3ArtifactStore("polymom", client, prefix="env")


@pytest.fixture(params=["local", "s3"])
def store(request: pytest.FixtureRequest) -> ArtifactStore:
    return request.getfixturevalue(f"{request.param}_store")  # type: ignore[no-any-return]


async def test_put_get_stream_exists_delete(store: ArtifactStore, tmp_path: Path) -> None:
    key = "meetings/m1/r1/exports/minutes.md"
    assert not await store.exists(key)

    await store.put(key, "नमस्ते ଆମେ".encode(), "text/markdown")

    assert await store.exists(key)
    assert (await store.get(key)).decode() == "नमस्ते ଆମେ"
    assert b"".join([c async for c in store.stream(key)]) == "नमस्ते ଆମେ".encode()
    await store.delete(key)
    await store.delete(key)  # idempotent
    assert not await store.exists(key)
    with pytest.raises(FileNotFoundError):
        await store.get(key)
    with pytest.raises(FileNotFoundError):
        [c async for c in store.stream(key)]


async def test_put_file_and_local_path(store: ArtifactStore, tmp_path: Path) -> None:
    source = tmp_path / "in.wav"
    source.write_bytes(b"RIFF" * 1000)

    await store.put_file("meetings/m1/upload/original.wav", source, "audio/wav")

    async with store.local_path("meetings/m1/upload/original.wav") as path:
        assert path.suffix == ".wav"
        assert path.read_bytes() == b"RIFF" * 1000


async def test_delete_prefix(store: ArtifactStore) -> None:
    meeting = uuid.uuid4()
    run = uuid.uuid4()
    keys = [run_key(meeting, run, "processed.wav"), f"{meeting_prefix(meeting)}upload/original.wav"]
    for key in keys:
        await store.put(key, b"x")
    await store.put("meetings/other/keep.txt", b"y")

    deleted = await store.delete_prefix(meeting_prefix(meeting))

    assert sorted(deleted) == sorted(keys)
    assert not any([await store.exists(k) for k in keys])
    assert await store.exists("meetings/other/keep.txt")
    assert await store.delete_prefix("meetings/none/") == []


async def test_presigned_urls(local_store: LocalArtifactStore, s3_store: S3ArtifactStore) -> None:
    assert await local_store.presigned_url("meetings/a.pdf") is None
    await s3_store.put("meetings/a.pdf", b"%PDF")
    url = await s3_store.presigned_url("meetings/a.pdf", 60)
    assert url is not None
    assert "env/meetings/a.pdf" in url
    assert "Signature=" in url or "X-Amz-Signature=" in url


@pytest.mark.parametrize("key", ["", "/etc/passwd", "meetings/../../x", "a b", "meetings\\x"])
def test_rejects_unsafe_keys(key: str) -> None:
    with pytest.raises(ValueError, match="invalid artifact key"):
        validate_key(key)


async def test_local_put_file_onto_itself_is_a_no_op(local_store: LocalArtifactStore) -> None:
    await local_store.put("a/b.bin", b"data")
    await local_store.put_file("a/b.bin", local_store.path("a/b.bin"))
    assert await local_store.get("a/b.bin") == b"data"
    assert await local_store.delete_prefix("a/b.bin") == ["a/b.bin"]
