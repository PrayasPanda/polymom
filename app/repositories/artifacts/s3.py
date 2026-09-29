"""Artifacts in an S3 bucket (AWS S3, MinIO, ...) via boto3 from the optional ``s3`` extra.

boto3 is synchronous, so every call runs in the threadpool.
"""

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from starlette.concurrency import run_in_threadpool

from app.repositories.artifacts.base import CHUNK_BYTES, ArtifactStore, validate_key


class S3ArtifactStore(ArtifactStore):
    backend = "s3"

    def __init__(self, bucket: str, client: Any, prefix: str = "") -> None:
        self.bucket = bucket
        self.client = client
        self.prefix = prefix.strip("/") + "/" if prefix.strip("/") else ""

    @classmethod
    def from_settings(
        cls,
        bucket: str,
        endpoint_url: str | None,
        access_key: str | None,
        secret_key: str | None,
        region: str | None,
    ) -> "S3ArtifactStore":  # pragma: no cover - exercised against real S3/MinIO
        import boto3

        client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region,
        )
        return cls(bucket, client)

    def _key(self, key: str) -> str:
        return self.prefix + validate_key(key)

    async def put(self, key: str, data: bytes, content_type: str | None = None) -> None:
        extra = {"ContentType": content_type} if content_type else {}
        await run_in_threadpool(
            self.client.put_object, Bucket=self.bucket, Key=self._key(key), Body=data, **extra
        )

    async def put_file(self, key: str, path: Path, content_type: str | None = None) -> None:
        extra = {"ExtraArgs": {"ContentType": content_type}} if content_type else {}
        await run_in_threadpool(
            self.client.upload_file, str(path), self.bucket, self._key(key), **extra
        )

    async def get(self, key: str) -> bytes:
        try:
            response = await run_in_threadpool(
                self.client.get_object, Bucket=self.bucket, Key=self._key(key)
            )
        except self.client.exceptions.NoSuchKey as exc:
            raise FileNotFoundError(key) from exc
        data: bytes = await run_in_threadpool(response["Body"].read)
        return data

    async def stream(self, key: str) -> AsyncIterator[bytes]:
        try:
            response = await run_in_threadpool(
                self.client.get_object, Bucket=self.bucket, Key=self._key(key)
            )
        except self.client.exceptions.NoSuchKey as exc:
            raise FileNotFoundError(key) from exc
        body = response["Body"]
        while chunk := await run_in_threadpool(body.read, CHUNK_BYTES):
            yield chunk

    async def exists(self, key: str) -> bool:
        try:
            await run_in_threadpool(self.client.head_object, Bucket=self.bucket, Key=self._key(key))
        except self.client.exceptions.ClientError:
            return False
        return True

    async def delete(self, key: str) -> None:
        await run_in_threadpool(self.client.delete_object, Bucket=self.bucket, Key=self._key(key))

    async def delete_prefix(self, prefix: str) -> list[str]:
        paginator = self.client.get_paginator("list_objects_v2")

        def _delete() -> list[str]:
            deleted: list[str] = []
            for page in paginator.paginate(Bucket=self.bucket, Prefix=self._key(prefix)):
                keys = [{"Key": o["Key"]} for o in page.get("Contents", [])]
                if keys:
                    self.client.delete_objects(Bucket=self.bucket, Delete={"Objects": keys})
                    deleted += [k["Key"][len(self.prefix) :] for k in keys]
            return deleted

        return await run_in_threadpool(_delete)

    async def presigned_url(self, key: str, expires_seconds: int = 3600) -> str | None:
        url: str = await run_in_threadpool(
            self.client.generate_presigned_url,
            "get_object",
            Params={"Bucket": self.bucket, "Key": self._key(key)},
            ExpiresIn=expires_seconds,
        )
        return url
