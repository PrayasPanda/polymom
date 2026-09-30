"""API key storage. Only the SHA-256 hex digest is stored; the plaintext leaves once."""

import hashlib
import secrets
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.base import utcnow
from app.models.api_key import ApiKey, IdempotencyRecord

KEY_PREFIX = "pk_"


def hash_key(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def generate_key() -> str:
    """A 40-character URL-safe token, prefixed for grep-ability."""
    return KEY_PREFIX + secrets.token_urlsafe(30)


class ApiKeyRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(
        self, *, label: str, owner: str | None, scopes: str = ""
    ) -> tuple[ApiKey, str]:
        token = generate_key()
        record = ApiKey(
            key_hash=hash_key(token),
            prefix=token[:12],
            label=label,
            owner=owner,
            scopes=scopes,
            active=True,
        )
        self._session.add(record)
        await self._session.flush()
        return record, token

    async def find_active(self, token: str) -> ApiKey | None:
        record = await self._session.scalar(
            select(ApiKey).where(ApiKey.key_hash == hash_key(token), ApiKey.active.is_(True))
        )
        if record is not None:
            record.last_used_at = utcnow()
        return record

    async def find_by_prefix(self, prefix: str) -> ApiKey | None:
        return await self._session.scalar(select(ApiKey).where(ApiKey.prefix == prefix))

    async def list_all(self) -> list[ApiKey]:
        return list(await self._session.scalars(select(ApiKey).order_by(ApiKey.id)))

    async def deactivate(self, prefix: str) -> bool:
        record = await self.find_by_prefix(prefix)
        if record is None:
            return False
        record.active = False
        await self._session.flush()
        return True


class IdempotencyRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get(self, api_key_id: int, idempotency_key: str) -> IdempotencyRecord | None:
        return await self._session.scalar(
            select(IdempotencyRecord).where(
                IdempotencyRecord.api_key_id == api_key_id,
                IdempotencyRecord.idempotency_key == idempotency_key,
            )
        )

    async def save(
        self,
        *,
        api_key_id: int,
        idempotency_key: str,
        method: str,
        path: str,
        status_code: int,
        response_body: str,
        meeting_id: uuid.UUID | None,
    ) -> IdempotencyRecord:
        record = IdempotencyRecord(
            api_key_id=api_key_id,
            idempotency_key=idempotency_key,
            method=method,
            path=path,
            status_code=status_code,
            response_body=response_body[:8192],
            meeting_id=meeting_id,
        )
        self._session.add(record)
        await self._session.flush()
        return record
