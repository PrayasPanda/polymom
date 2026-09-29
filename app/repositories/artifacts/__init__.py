"""Artifact stores selected by ``ARTIFACT_STORE=local|s3``."""

from functools import cache

from app.core.config import Settings
from app.repositories.artifacts.base import ArtifactStore, meeting_prefix, run_key
from app.repositories.artifacts.local import LocalArtifactStore

__all__ = [
    "ArtifactStore",
    "LocalArtifactStore",
    "build_artifact_store",
    "meeting_prefix",
    "run_key",
]


def build_artifact_store(settings: Settings) -> ArtifactStore:
    if settings.artifact_store == "s3":  # pragma: no cover - needs S3/MinIO
        return _s3_store(
            settings.s3_bucket,
            settings.s3_endpoint_url,
            settings.s3_access_key.get_secret_value() if settings.s3_access_key else None,
            settings.s3_secret_key.get_secret_value() if settings.s3_secret_key else None,
            settings.s3_region,
        )
    return LocalArtifactStore(settings.storage_dir)


@cache
def _s3_store(
    bucket: str,
    endpoint_url: str | None,
    access_key: str | None,
    secret_key: str | None,
    region: str | None,
) -> ArtifactStore:  # pragma: no cover - needs S3/MinIO
    from app.repositories.artifacts.s3 import S3ArtifactStore

    return S3ArtifactStore.from_settings(bucket, endpoint_url, access_key, secret_key, region)
