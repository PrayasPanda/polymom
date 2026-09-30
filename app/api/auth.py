"""API-key authentication (``X-API-Key`` header), with optional dev-mode bypass."""

from typing import Annotated

import structlog
from fastapi import Depends, Header, Request

from app.api.deps import SettingsDep, UowDep
from app.core.config import Settings
from app.core.exceptions import AuthenticationError
from app.models.api_key import ApiKey

API_KEY_HEADER = "X-API-Key"
INTERNAL_OWNER_ID = 0  # sentinel for the dev/test bypass


async def resolve_api_key(
    request: Request,
    uow: UowDep,
    settings: SettingsDep,
    x_api_key: Annotated[str | None, Header(alias=API_KEY_HEADER)] = None,
) -> ApiKey | None:
    """Return the caller's API key, or ``None`` when auth is disabled.

    When ``API_KEY_REQUIRED=false`` (the default in tests) a missing key is fine;
    a valid one is still resolved so its owner scoping applies.
    """
    if x_api_key:
        record = await uow.api_keys.find_active(x_api_key)
        if record is None:
            raise AuthenticationError("Unknown or inactive API key.")
        await uow.commit()  # persist last_used_at
        structlog.contextvars.bind_contextvars(api_key_id=record.id)
        request.state.api_key_id = record.id
        return record
    if settings.api_key_required:
        raise AuthenticationError("An API key is required. Set the X-API-Key header.")
    return None


CurrentApiKey = Annotated[ApiKey | None, Depends(resolve_api_key)]


def current_owner_id(key: ApiKey | None, settings: Settings) -> int | None:
    """The ``owner_key_id`` to apply to database queries. ``None`` = unscoped (dev)."""
    if key is not None:
        return key.id
    return None if not settings.api_key_required else INTERNAL_OWNER_ID


async def owner_id_dependency(
    key: CurrentApiKey,
    settings: SettingsDep,
    uow: UowDep,
) -> int | None:
    return current_owner_id(key, settings)


OwnerIdDep = Annotated[int | None, Depends(owner_id_dependency)]
