"""Dependency-injection providers for FastAPI routes."""

from functools import lru_cache
from typing import Annotated

from fastapi import Depends

from app.core.config import Settings, get_settings
from app.repositories.meeting_repository import InMemoryMeetingRepository, MeetingRepository


@lru_cache
def get_meeting_repository() -> MeetingRepository:
    """Return the meeting repository. TODO: swap for a durable backend."""
    return InMemoryMeetingRepository()


SettingsDep = Annotated[Settings, Depends(get_settings)]
MeetingRepoDep = Annotated[MeetingRepository, Depends(get_meeting_repository)]
