"""Consolidated results, exports, utterance queries, run history and search."""

import uuid
from collections.abc import Sequence

from starlette.concurrency import run_in_threadpool

from app.models.results import ProcessingRun
from app.repositories.artifacts import run_key
from app.repositories.results_repository import UtteranceQuery
from app.schemas.analytics import ConversationAnalytics, SpeakerStats
from app.schemas.audio import AudioQuality
from app.schemas.language import LanguageSummary
from app.schemas.result import (
    SECTIONS,
    MeetingResult,
    ProcessingInfo,
    RunList,
    RunSummary,
    SearchHit,
    SearchResults,
    Section,
    SpeakerInfo,
    StageInfo,
    UtterancePage,
    UtteranceRow,
)
from app.schemas.summary import MeetingSummary
from app.schemas.transcript import SpeakerTranscript
from app.services.analytics.charts import ChartName, render_chart
from app.services.export import ExportFormat, render_export
from app.services.meeting_service import MeetingService
from app.services.stage_outputs import load_stage_output

EXPORT_EXTENSIONS = {"docx": "docx", "pdf": "pdf", "md": "md", "json": "json"}


class ResultService:
    def __init__(self, meetings: MeetingService) -> None:
        self.meetings = meetings
        self.uow = meetings.uow
        self.store = meetings.store

    async def _output(self, run: ProcessingRun | None, stage: str) -> object | None:
        if run is None:
            return None
        return await load_stage_output(self.uow.results, self.store, run.id, stage)

    async def result(
        self,
        meeting_id: uuid.UUID,
        run_id: uuid.UUID | None = None,
        include: Sequence[Section] = SECTIONS,
    ) -> MeetingResult:
        meeting = await self.meetings.get(meeting_id)
        run = await self.meetings.run_for(meeting_id, run_id)
        speakers = await self.uow.results.speakers(meeting_id)
        names = {s.label: s.display_name for s in speakers if s.display_name}

        warnings: list[str] = []
        processing = None
        quality = transcript = analytics = summary = languages = None
        if run is not None:
            stages = await self.uow.results.stage_results(run.id)
            processing = ProcessingInfo(
                run_id=run.id,
                status=run.status,
                started_at=run.started_at,
                finished_at=run.finished_at,
                model_versions=run.model_versions or {},
                timings_ms=run.timings_ms or {},
                stages=[
                    StageInfo(
                        name=s.stage_name, status=s.status, duration_ms=s.duration_ms, error=s.error
                    )
                    for s in stages
                ],
                error=run.error,
            )
            warnings += [
                f"{s.stage_name} failed: {s.error}" for s in stages if s.status == "failed"
            ]
            raw_quality = await self._output(run, "preprocess")
            quality = AudioQuality.model_validate(raw_quality) if raw_quality else None
            if quality:
                warnings += [w.message for w in quality.warnings]
            raw_langs = await self._output(run, "identify_languages")
            languages = LanguageSummary.model_validate(raw_langs) if raw_langs else None
            if "transcript" in include and (raw := await self._output(run, "align")):
                transcript = SpeakerTranscript.model_validate(raw)
                for u in transcript.utterances:
                    u.speaker_name = names.get(u.speaker)
                warnings += transcript.warnings
            if "analytics" in include and (raw := await self._output(run, "analytics")):
                analytics = ConversationAnalytics.model_validate(raw)
                for s in analytics.speakers:
                    s.speaker_name = names.get(s.speaker)
            if "summary" in include and (record := await self.uow.results.latest_summary(run.id)):
                summary = MeetingSummary.model_validate(record.content)

        return MeetingResult(
            meeting=await self.meetings.to_read(meeting),
            processing=processing,
            audio_quality=quality,
            languages=languages,
            speakers=[
                SpeakerInfo(
                    label=s.label,
                    display_name=s.display_name,
                    stats=SpeakerStats.model_validate(s.stats) if s.stats else None,
                )
                for s in speakers
            ],
            transcript=transcript,
            analytics=analytics,
            summary=summary,
            verification_report=summary.verification_report if summary else None,
            warnings=warnings,
            included=[s for s in SECTIONS if s in include],
        )

    # --- exports ---

    async def export(
        self, meeting_id: uuid.UUID, fmt: ExportFormat, run_id: uuid.UUID | None = None
    ) -> tuple[bytes, uuid.UUID | None]:
        """Rendered minutes, cached per run and format in the artifact store."""
        run = await self.meetings.run_for(meeting_id, run_id)
        key = (
            run_key(meeting_id, run.id, f"exports/minutes.{EXPORT_EXTENSIONS[fmt]}")
            if run
            else None
        )
        if key and await self.store.exists(key):
            return await self.store.get(key), run.id if run else None
        result = await self.result(meeting_id, run.id if run else None)
        data = await run_in_threadpool(render_export, fmt, result)
        if key:
            await self.store.put(key, data)
        return data, run.id if run else None

    async def chart(self, meeting_id: uuid.UUID, name: ChartName) -> bytes:
        """PNG chart of the default run, cached in the artifact store."""
        analytics, names = await self.meetings.get_analytics(meeting_id)
        run = await self.meetings.run_for(meeting_id)
        key = run_key(meeting_id, run.id, f"charts/{name}.png") if run else None
        if key and await self.store.exists(key):
            return await self.store.get(key)
        turns = (
            (await self.meetings.get_diarization(meeting_id)).turns if name == "timeline" else []
        )
        png = await run_in_threadpool(render_chart, name, analytics, names, turns)
        if key:
            await self.store.put(key, png, "image/png")
        return png

    # --- utterances, runs, search ---

    async def utterances(
        self, meeting_id: uuid.UUID, query: UtteranceQuery, run_id: uuid.UUID | None = None
    ) -> UtterancePage | None:
        run = await self.meetings.run_for(meeting_id, run_id)
        if run is None:
            return None
        rows, total = await self.uow.results.query_utterances(run.id, query)
        names = await self.meetings.speaker_names(meeting_id)
        return UtterancePage(
            meeting_id=meeting_id,
            run_id=run.id,
            items=[
                UtteranceRow(
                    id=r.utterance_index,
                    speaker=r.speaker,
                    speaker_name=names.get(r.speaker),
                    start=r.start,
                    end=r.end,
                    text=r.text,
                    primary_language=r.primary_language,
                    is_code_mixed=r.is_code_mixed,
                    has_overlap=r.has_overlap,
                    alignment_precision=r.alignment_precision,
                )
                for r in rows
            ],
            total=total,
            limit=query.limit,
            offset=query.offset,
        )

    async def runs(self, meeting_id: uuid.UUID) -> RunList:
        await self.meetings.get(meeting_id)
        default = await self.meetings.run_for(meeting_id)
        return RunList(
            meeting_id=meeting_id,
            items=[
                RunSummary(
                    run_id=r.id,
                    status=r.status,
                    started_at=r.started_at,
                    finished_at=r.finished_at,
                    model_versions=r.model_versions or {},
                    error=r.error,
                    is_default=default is not None and r.id == default.id,
                )
                for r in await self.uow.results.list_runs(meeting_id)
            ],
        )

    async def search(
        self, q: str, limit: int, meeting_id: uuid.UUID | None = None
    ) -> SearchResults:
        rows = await self.uow.search.search(q, limit=limit, meeting_id=meeting_id)
        titles: dict[uuid.UUID, str | None] = {}
        names: dict[uuid.UUID, dict[str, str]] = {}
        hits = []
        for r in rows:
            if r.meeting_id not in titles:
                meeting = await self.uow.meetings.get(r.meeting_id)
                titles[r.meeting_id] = meeting.title if meeting else None
                names[r.meeting_id] = await self.meetings.speaker_names(r.meeting_id)
            hits.append(
                SearchHit(
                    kind=r.kind,
                    meeting_id=r.meeting_id,
                    meeting_title=titles[r.meeting_id],
                    run_id=r.run_id,
                    utterance_id=int(r.ref) if r.kind == "utterance" else None,
                    speaker=r.speaker,
                    speaker_name=names[r.meeting_id].get(r.speaker) if r.speaker else None,
                    start=r.start,
                    text=r.text,
                )
            )
        return SearchResults(query=q, items=hits)
