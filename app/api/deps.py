"""Dependency-injection providers for FastAPI routes."""

from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import Settings, get_settings
from app.pipelines.mom_pipeline import (
    MoMPipeline,
    SummarizationStage,
    SummaryRegenerator,
    build_pipeline,
)
from app.repositories.artifacts import ArtifactStore, build_artifact_store
from app.repositories.unit_of_work import UnitOfWork, UnitOfWorkFactory, unit_of_work_factory
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
from app.services.result_service import ResultService

SettingsDep = Annotated[Settings, Depends(get_settings)]


def get_uow_factory(request: Request) -> UnitOfWorkFactory:
    """Short-lived units of work for work that outlives the request (background tasks)."""
    sessionmaker: async_sessionmaker[AsyncSession] = request.app.state.sessionmaker
    return unit_of_work_factory(sessionmaker)


UowFactoryDep = Annotated[UnitOfWorkFactory, Depends(get_uow_factory)]


async def get_uow(factory: UowFactoryDep) -> AsyncIterator[UnitOfWork]:
    async with factory() as uow:
        yield uow


UowDep = Annotated[UnitOfWork, Depends(get_uow)]


def get_artifact_store(settings: SettingsDep) -> ArtifactStore:
    """Chosen by ``ARTIFACT_STORE`` (local files under STORAGE_DIR, or S3/MinIO)."""
    return build_artifact_store(settings)


ArtifactStoreDep = Annotated[ArtifactStore, Depends(get_artifact_store)]


def get_meeting_service(
    uow: UowDep,
    settings: SettingsDep,
    store: ArtifactStoreDep,
    request: Request,
) -> MeetingService:
    owner_key_id = getattr(request.state, "api_key_id", None)
    return MeetingService(uow, MediaValidator(settings), settings, store, owner_key_id)


MeetingServiceDep = Annotated[MeetingService, Depends(get_meeting_service)]


def get_result_service(meetings: MeetingServiceDep) -> ResultService:
    return ResultService(meetings)


ResultServiceDep = Annotated[ResultService, Depends(get_result_service)]


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
    settings: SettingsDep,
    uow_factory: UowFactoryDep,
    store: ArtifactStoreDep,
    diarization_backend: Annotated[DiarizationBackend, Depends(get_diarization_backend)],
    asr_router: Annotated[ASRRouter, Depends(get_asr_router)],
    language_identifier: Annotated[LanguageIdentifier, Depends(get_language_identifier)],
    llm: LLMClientDep,
) -> MoMPipeline:
    return build_pipeline(
        settings, uow_factory, store, diarization_backend, asr_router, language_identifier, llm
    )


PipelineDep = Annotated[MoMPipeline, Depends(get_pipeline)]


def get_summary_regenerator(
    settings: SettingsDep, uow_factory: UowFactoryDep, store: ArtifactStoreDep, llm: LLMClientDep
) -> SummaryRegenerator:
    return SummaryRegenerator(SummarizationStage(settings, llm), uow_factory, store, settings)


SummaryRegeneratorDep = Annotated[SummaryRegenerator, Depends(get_summary_regenerator)]
