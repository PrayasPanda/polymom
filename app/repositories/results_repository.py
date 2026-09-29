"""Runs, stage results, speakers, utterances and summaries."""

import uuid
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.base import utcnow
from app.models.results import (
    SUCCESSFUL_RUN_STATUSES,
    ArtifactCleanup,
    ProcessingRun,
    Speaker,
    StageResult,
    SummaryRecord,
    UtteranceRecord,
)
from app.schemas.summary import MeetingSummary
from app.schemas.transcript import Utterance


@dataclass
class UtteranceQuery:
    speaker: str | None = None
    language: str | None = None
    start_from: float | None = None
    end_to: float | None = None
    q: str | None = None
    limit: int = 100
    offset: int = 0


class ResultsRepository(ABC):
    @abstractmethod
    async def create_run(
        self, meeting_id: uuid.UUID, config_snapshot: dict[str, Any]
    ) -> ProcessingRun: ...

    @abstractmethod
    async def get_run(self, meeting_id: uuid.UUID, run_id: uuid.UUID) -> ProcessingRun | None: ...

    @abstractmethod
    async def list_runs(self, meeting_id: uuid.UUID) -> list[ProcessingRun]:
        """Newest first."""

    @abstractmethod
    async def latest_run(
        self, meeting_id: uuid.UUID, *, successful: bool = True
    ) -> ProcessingRun | None: ...

    @abstractmethod
    async def save_stage(
        self,
        run_id: uuid.UUID,
        stage_name: str,
        *,
        status: str,
        output: Any = None,
        output_ref: str | None = None,
        duration_ms: int | None = None,
        error: str | None = None,
    ) -> StageResult:
        """Insert or replace the result of one stage of a run."""

    @abstractmethod
    async def stage_result(self, run_id: uuid.UUID, stage_name: str) -> StageResult | None: ...

    @abstractmethod
    async def stage_results(self, run_id: uuid.UUID) -> list[StageResult]: ...

    @abstractmethod
    async def replace_utterances(
        self, meeting_id: uuid.UUID, run_id: uuid.UUID, utterances: Sequence[Utterance]
    ) -> None: ...

    @abstractmethod
    async def query_utterances(
        self, run_id: uuid.UUID, query: UtteranceQuery
    ) -> tuple[list[UtteranceRecord], int]: ...

    @abstractmethod
    async def speakers(self, meeting_id: uuid.UUID) -> list[Speaker]: ...

    @abstractmethod
    async def upsert_speakers(
        self,
        meeting_id: uuid.UUID,
        labels: Sequence[str],
        stats: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        """Ensure a row per label; replace stats when given. Display names are kept."""

    @abstractmethod
    async def set_display_names(
        self, meeting_id: uuid.UUID, names: dict[str, str | None]
    ) -> dict[str, str]: ...

    @abstractmethod
    async def add_summary(
        self, meeting_id: uuid.UUID, run_id: uuid.UUID, summary: MeetingSummary
    ) -> SummaryRecord: ...

    @abstractmethod
    async def latest_summary(self, run_id: uuid.UUID) -> SummaryRecord | None: ...

    @abstractmethod
    async def log_cleanup(self, key: str, error: str) -> None: ...

    @abstractmethod
    async def pending_cleanups(self) -> list[ArtifactCleanup]: ...

    @abstractmethod
    async def resolve_cleanup(self, entry: ArtifactCleanup) -> None: ...


class SqlAlchemyResultsRepository(ResultsRepository):
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create_run(
        self, meeting_id: uuid.UUID, config_snapshot: dict[str, Any]
    ) -> ProcessingRun:
        run = ProcessingRun(
            id=uuid.uuid4(),
            meeting_id=meeting_id,
            status="processing",
            started_at=utcnow(),
            config_snapshot=config_snapshot,
            model_versions={},
            timings_ms={},
        )
        self._session.add(run)
        await self._session.flush()
        return run

    async def get_run(self, meeting_id: uuid.UUID, run_id: uuid.UUID) -> ProcessingRun | None:
        run = await self._session.get(ProcessingRun, run_id)
        return run if run is not None and run.meeting_id == meeting_id else None

    async def list_runs(self, meeting_id: uuid.UUID) -> list[ProcessingRun]:
        return list(
            await self._session.scalars(
                select(ProcessingRun)
                .where(ProcessingRun.meeting_id == meeting_id)
                .order_by(ProcessingRun.started_at.desc())
            )
        )

    async def latest_run(
        self, meeting_id: uuid.UUID, *, successful: bool = True
    ) -> ProcessingRun | None:
        stmt = select(ProcessingRun).where(ProcessingRun.meeting_id == meeting_id)
        if successful:
            stmt = stmt.where(ProcessingRun.status.in_(SUCCESSFUL_RUN_STATUSES))
        return await self._session.scalar(stmt.order_by(ProcessingRun.started_at.desc()).limit(1))

    async def save_stage(
        self,
        run_id: uuid.UUID,
        stage_name: str,
        *,
        status: str,
        output: Any = None,
        output_ref: str | None = None,
        duration_ms: int | None = None,
        error: str | None = None,
    ) -> StageResult:
        existing = await self.stage_result(run_id, stage_name)
        row = existing or StageResult(run_id=run_id, stage_name=stage_name)
        row.status = status
        row.output = output
        row.output_ref = output_ref
        row.duration_ms = duration_ms
        row.error = error
        row.created_at = utcnow()
        if existing is None:
            self._session.add(row)
        await self._session.flush()
        return row

    async def stage_result(self, run_id: uuid.UUID, stage_name: str) -> StageResult | None:
        return await self._session.scalar(
            select(StageResult).where(
                StageResult.run_id == run_id, StageResult.stage_name == stage_name
            )
        )

    async def stage_results(self, run_id: uuid.UUID) -> list[StageResult]:
        return list(
            await self._session.scalars(
                select(StageResult).where(StageResult.run_id == run_id).order_by(StageResult.id)
            )
        )

    async def replace_utterances(
        self, meeting_id: uuid.UUID, run_id: uuid.UUID, utterances: Sequence[Utterance]
    ) -> None:
        await self._session.execute(delete(UtteranceRecord).where(UtteranceRecord.run_id == run_id))
        self._session.add_all(
            UtteranceRecord(
                meeting_id=meeting_id,
                run_id=run_id,
                utterance_index=u.id,
                speaker=u.speaker,
                start=u.start,
                end=u.end,
                text=u.text,
                primary_language=u.primary_language,
                is_code_mixed=u.is_code_mixed,
                has_overlap=u.has_overlap,
                alignment_precision=u.alignment_precision,
            )
            for u in utterances
        )
        await self._session.flush()

    async def query_utterances(
        self, run_id: uuid.UUID, query: UtteranceQuery
    ) -> tuple[list[UtteranceRecord], int]:
        filters = [UtteranceRecord.run_id == run_id]
        if query.speaker:
            filters.append(UtteranceRecord.speaker == query.speaker)
        if query.language:
            filters.append(UtteranceRecord.primary_language == query.language)
        if query.start_from is not None:
            filters.append(UtteranceRecord.end >= query.start_from)
        if query.end_to is not None:
            filters.append(UtteranceRecord.start <= query.end_to)
        if query.q:
            pattern = f"%{query.q.lower()}%"
            filters.append(func.lower(UtteranceRecord.text).like(pattern))
        total = await self._session.scalar(
            select(func.count()).select_from(UtteranceRecord).where(*filters)
        )
        rows = await self._session.scalars(
            select(UtteranceRecord)
            .where(*filters)
            .order_by(UtteranceRecord.start, UtteranceRecord.utterance_index)
            .limit(query.limit)
            .offset(query.offset)
        )
        return list(rows), total or 0

    async def speakers(self, meeting_id: uuid.UUID) -> list[Speaker]:
        return list(
            await self._session.scalars(
                select(Speaker).where(Speaker.meeting_id == meeting_id).order_by(Speaker.label)
            )
        )

    async def upsert_speakers(
        self,
        meeting_id: uuid.UUID,
        labels: Sequence[str],
        stats: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        existing = {s.label: s for s in await self.speakers(meeting_id)}
        for label in labels:
            row = existing.get(label)
            if row is None:
                row = Speaker(meeting_id=meeting_id, label=label)
                self._session.add(row)
            if stats is not None:
                row.stats = stats.get(label)
        await self._session.flush()

    async def set_display_names(
        self, meeting_id: uuid.UUID, names: dict[str, str | None]
    ) -> dict[str, str]:
        for label, name in names.items():
            await self._session.execute(
                update(Speaker)
                .where(Speaker.meeting_id == meeting_id, Speaker.label == label)
                .values(display_name=name, updated_at=utcnow())
            )
        await self._session.flush()
        return {s.label: s.display_name for s in await self.speakers(meeting_id) if s.display_name}

    async def add_summary(
        self, meeting_id: uuid.UUID, run_id: uuid.UUID, summary: MeetingSummary
    ) -> SummaryRecord:
        content = summary.model_dump(mode="json")
        record = SummaryRecord(
            run_id=run_id,
            meeting_id=meeting_id,
            output_language=summary.output_language,
            provider=summary.model_info.provider,
            model=summary.model_info.model,
            prompt_version=summary.model_info.prompt_version,
            content=content,
            verification_report=content["verification_report"],
            created_at=utcnow(),
        )
        self._session.add(record)
        await self._session.flush()
        return record

    async def latest_summary(self, run_id: uuid.UUID) -> SummaryRecord | None:
        return await self._session.scalar(
            select(SummaryRecord)
            .where(SummaryRecord.run_id == run_id)
            .order_by(SummaryRecord.created_at.desc(), SummaryRecord.id.desc())
            .limit(1)
        )

    async def log_cleanup(self, key: str, error: str) -> None:
        self._session.add(ArtifactCleanup(key=key, error=error[:2000], created_at=utcnow()))
        await self._session.flush()

    async def pending_cleanups(self) -> list[ArtifactCleanup]:
        return list(
            await self._session.scalars(select(ArtifactCleanup).order_by(ArtifactCleanup.id))
        )

    async def resolve_cleanup(self, entry: ArtifactCleanup) -> None:
        await self._session.delete(entry)
        await self._session.flush()
