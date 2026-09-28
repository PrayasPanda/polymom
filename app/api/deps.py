"""Dependency-injection providers for FastAPI routes."""

from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import Settings, get_settings
from app.pipelines.mom_pipeline import MoMPipeline, RepositoryFactory, build_pipeline
from app.repositories.meeting_repository import MeetingRepository, SqlAlchemyMeetingRepository
from app.services.audio.validator import MediaValidator
from app.services.diarization.base import DiarizationBackend
from app.services.diarization.service import build_backend
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


def repository_factory(
    sessionmaker: async_sessionmaker[AsyncSession],
) -> RepositoryFactory:
    """Short-lived repositories for work that outlives the request (background tasks)."""

    @asynccontextmanager
    async def _factory() -> AsyncIterator[MeetingRepository]:
        async with sessionmaker() as session:
            yield SqlAlchemyMeetingRepository(session)

    def _make() -> AbstractAsyncContextManager[MeetingRepository]:
        return _factory()

    return _make


def get_diarization_backend(settings: SettingsDep) -> DiarizationBackend:
    """Backend chosen by ``DIARIZATION_BACKEND`` (pyannote models are cached process-wide)."""
    return build_backend(settings)


def get_pipeline(
    request: Request,
    settings: SettingsDep,
    diarization_backend: Annotated[DiarizationBackend, Depends(get_diarization_backend)],
) -> MoMPipeline:
    return build_pipeline(
        settings, repository_factory(request.app.state.sessionmaker), diarization_backend
    )


PipelineDep = Annotated[MoMPipeline, Depends(get_pipeline)]
