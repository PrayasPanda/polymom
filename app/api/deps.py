"""Dependency-injection providers for FastAPI routes."""

from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import Settings, get_settings
from app.repositories.meeting_repository import MeetingRepository, SqlAlchemyMeetingRepository
from app.services.audio.validator import MediaValidator
from app.services.meeting_service import MeetingService

SettingsDep = Annotated[Settings, Depends(get_settings)]


async def get_session(request: Request) -> AsyncIterator[AsyncSession]:
    """Yield a session from the sessionmaker created in the app lifespan."""
    sessionmaker: async_sessionmaker[AsyncSession] = request.app.state.sessionmaker
    async with sessionmaker() as session:
        yield session


SessionDep = Annotated[AsyncSession, Depends(get_session)]


def get_meeting_repository(session: SessionDep) -> MeetingRepository:
    return SqlAlchemyMeetingRepository(session)


def get_media_validator(settings: SettingsDep) -> MediaValidator:
    return MediaValidator(settings)


def get_meeting_service(
    repository: Annotated[MeetingRepository, Depends(get_meeting_repository)],
    validator: Annotated[MediaValidator, Depends(get_media_validator)],
    settings: SettingsDep,
) -> MeetingService:
    return MeetingService(repository, validator, settings)


MeetingServiceDep = Annotated[MeetingService, Depends(get_meeting_service)]
