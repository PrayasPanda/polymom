"""Dependency-injection providers for FastAPI routes."""

from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import Settings, get_settings
from app.pipelines.mom_pipeline import (
    MoMPipeline,
    RepositoryFactory,
    SummarizationStage,
    SummaryRegenerator,
    build_pipeline,
)
from app.repositories.meeting_repository import MeetingRepository, SqlAlchemyMeetingRepository
from app.services.asr.router import ASRRouter
from app.services.asr.service import build_router
from app.services.audio.validator import MediaValidator
from app.services.diarization.base import DiarizationBackend
from app.services.diarization.service import build_backend
from app.services.language.base import LanguageIdentifier
from app.services.language.service import build_identifier
from app.services.llm import build_llm_client
from app.services.llm.base import LLMClient
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


def get_asr_router(settings: SettingsDep) -> ASRRouter:
    """Backends chosen by ``ASR_BACKEND``, routed per ``ASR_LANGUAGE_BACKENDS``."""
    return build_router(settings)


def get_language_identifier(settings: SettingsDep) -> LanguageIdentifier:
    """Spoken language ID chosen by ``LID_BACKEND``."""
    return build_identifier(settings)


def get_llm_client(settings: SettingsDep) -> LLMClient:
    """LLM chosen by ``LLM_PROVIDER`` / ``LLM_MODEL``."""
    return build_llm_client(settings)


LLMClientDep = Annotated[LLMClient, Depends(get_llm_client)]


def get_pipeline(
    request: Request,
    settings: SettingsDep,
    diarization_backend: Annotated[DiarizationBackend, Depends(get_diarization_backend)],
    asr_router: Annotated[ASRRouter, Depends(get_asr_router)],
    language_identifier: Annotated[LanguageIdentifier, Depends(get_language_identifier)],
    llm: LLMClientDep,
) -> MoMPipeline:
    return build_pipeline(
        settings,
        repository_factory(request.app.state.sessionmaker),
        diarization_backend,
        asr_router,
        language_identifier,
        llm,
    )


PipelineDep = Annotated[MoMPipeline, Depends(get_pipeline)]


def get_summary_regenerator(
    request: Request, settings: SettingsDep, llm: LLMClientDep
) -> SummaryRegenerator:
    return SummaryRegenerator(
        SummarizationStage(settings, llm), repository_factory(request.app.state.sessionmaker)
    )


SummaryRegeneratorDep = Annotated[SummaryRegenerator, Depends(get_summary_regenerator)]
