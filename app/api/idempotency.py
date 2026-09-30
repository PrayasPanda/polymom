"""``Idempotency-Key`` support for POST endpoints that create work (upload, process).

The first response for a (API key, Idempotency-Key) pair is stored; a retry with the
same key returns that response unchanged (``Idempotent-Replayed: true``) instead of
uploading or enqueueing again. Reusing a key for a different endpoint is a 409.
"""

import json
import uuid

from fastapi import Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.core.exceptions import IdempotencyConflictError, ValidationError
from app.repositories.unit_of_work import UnitOfWork

MAX_KEY_LENGTH = 200
ANONYMOUS_KEY_ID = 0  # when API keys are disabled (tests, local dev)


def _owner(request: Request) -> int:
    return int(getattr(request.state, "api_key_id", None) or ANONYMOUS_KEY_ID)


def _check(key: str) -> None:
    if not key or len(key) > MAX_KEY_LENGTH or not key.isprintable():
        raise ValidationError(
            "Idempotency-Key must be 1-200 printable characters.",
            details={"max_length": MAX_KEY_LENGTH},
        )


async def replay(uow: UnitOfWork, request: Request, key: str | None) -> JSONResponse | None:
    """The stored response for this key, or ``None`` if the request is new."""
    if not key:
        return None
    _check(key)
    record = await uow.idempotency.get(_owner(request), key)
    if record is None:
        return None
    if record.method != request.method or record.path != request.url.path:
        raise IdempotencyConflictError(
            "This Idempotency-Key was already used for a different request.",
            details={"original": f"{record.method} {record.path}"},
        )
    return JSONResponse(
        json.loads(record.response_body),
        status_code=record.status_code,
        headers={"Idempotent-Replayed": "true"},
    )


async def remember(
    uow: UnitOfWork,
    request: Request,
    key: str | None,
    status_code: int,
    body: BaseModel,
    meeting_id: uuid.UUID | None,
) -> None:
    if not key:
        return
    await uow.idempotency.save(
        api_key_id=_owner(request),
        idempotency_key=key,
        method=request.method,
        path=request.url.path,
        status_code=status_code,
        response_body=body.model_dump_json(),
        meeting_id=meeting_id,
    )
    await uow.commit()
