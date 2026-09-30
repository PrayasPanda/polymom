"""Meeting endpoints: upload, process, speakers, transcript, analytics, summary, results."""

from datetime import datetime
from typing import Annotated, Literal
from uuid import UUID

from fastapi import (
    APIRouter,
    BackgroundTasks,
    File,
    Form,
    Header,
    Query,
    Request,
    Response,
    UploadFile,
    status,
)
from fastapi.responses import PlainTextResponse

from app.api.deps import (
    MeetingServiceDep,
    PipelineDep,
    ResultServiceDep,
    SettingsDep,
    SummaryRegeneratorDep,
)
from app.core.exceptions import TranscriptNotAvailableError, ValidationError
from app.repositories.meeting_repository import MeetingQuery, SortField
from app.repositories.results_repository import UtteranceQuery
from app.schemas.analytics import ConversationAnalyticsResponse
from app.schemas.asr import TranscriptResponse
from app.schemas.diarization import SpeakersResponse
from app.schemas.error import error_example
from app.schemas.language import LanguageSummaryResponse
from app.schemas.meeting import (
    SUPPORTED_LANGUAGES,
    MeetingCreateResponse,
    MeetingList,
    MeetingRead,
    MeetingStatus,
    ProcessResponse,
)
from app.schemas.result import SECTIONS, MeetingResult, RunList, Section, UtterancePage
from app.schemas.summary import (
    MeetingSummaryResponse,
    RegenerateSummaryRequest,
    RegenerateSummaryResponse,
)
from app.schemas.transcript import (
    SpeakerNamesResponse,
    SpeakerRenameRequest,
    SpeakerTranscriptResponse,
)
from app.services.analytics.charts import ChartName
from app.services.analytics.export import speakers_to_csv
from app.services.export import MEDIA_TYPES, ExportFormat
from app.services.summarization.render import summary_to_markdown
from app.utils.subtitles import (
    to_srt,
    to_text,
    utterances_to_markdown,
    utterances_to_srt,
    utterances_to_text,
    utterances_to_vtt,
)
from app.workers.queue import build_queue
from app.workers.webhook import validate_callback_url

router = APIRouter(prefix="/meetings", tags=["meetings"])

_NOT_FOUND = error_example(404, "meeting_not_found", "Meeting ... not found.", "Unknown meeting")


def parse_languages(values: list[str] | None) -> list[str]:
    """Accept repeated fields and/or comma-separated values; dedupe, keep order."""
    langs: list[str] = []
    for value in values or []:
        for lang in (v.strip().lower() for v in value.split(",")):
            if lang and lang not in langs:
                langs.append(lang)
    invalid = [lang for lang in langs if lang not in SUPPORTED_LANGUAGES]
    if invalid:
        raise ValidationError(
            "Unsupported language hint.",
            details={"invalid": invalid, "supported": sorted(SUPPORTED_LANGUAGES)},
        )
    return langs


@router.post(
    "",
    response_model=MeetingCreateResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload a meeting recording",
    responses={
        **error_example(
            413, "file_too_large", "File exceeds the 200 MB upload limit.", "Too large"
        ),
        **error_example(
            415,
            "unsupported_file_type",
            "File content does not match its extension.",
            "Extension not allowed, or content does not match the extension",
        ),
        **error_example(
            422,
            "corrupted_media",
            "The file contains no audio stream.",
            "Empty, corrupted or audio-less file, or invalid form fields",
        ),
    },
)
async def create_meeting(
    service: MeetingServiceDep,
    file: Annotated[UploadFile, File(description="Audio or video recording.")],
    title: Annotated[str | None, Form(max_length=200)] = None,
    expected_speakers: Annotated[
        int | None, Form(ge=1, le=20, description="Expected number of speakers (1-20).")
    ] = None,
    languages: Annotated[
        list[str] | None,
        Form(description="Language hints: en, hi, or. Repeat the field or comma-separate."),
    ] = None,
    allow_duplicate: Annotated[
        bool, Query(description="Store a new meeting even if the same file was uploaded.")
    ] = False,
    *,
    response: Response,
) -> MeetingCreateResponse:
    """Validate the upload, store it and queue the meeting for processing.

    Uploads are idempotent: the SHA-256 of an identical earlier file returns that
    meeting with ``200`` and ``duplicate: true`` unless ``?allow_duplicate=true``.
    """
    langs = parse_languages(languages)
    try:
        meeting, duplicate = await service.create(
            source=file,
            filename=file.filename,
            title=title.strip() if title and title.strip() else None,
            expected_speakers=expected_speakers,
            languages=langs,
            allow_duplicate=allow_duplicate,
        )
    finally:
        await file.close()
    if duplicate:
        response.status_code = status.HTTP_200_OK
    return MeetingCreateResponse(
        meeting_id=meeting.id,
        status=meeting.status,
        created_at=meeting.created_at,
        duplicate=duplicate,
    )


@router.get(
    "",
    response_model=MeetingList,
    summary="List meetings with filters, sorting and cursor pagination",
    responses=error_example(422, "validation_error", "Invalid cursor.", "Bad cursor or filter"),
)
async def list_meetings(
    service: MeetingServiceDep,
    status_: Annotated[MeetingStatus | None, Query(alias="status")] = None,
    language: Annotated[
        str | None, Query(description="Spoken language detected in the meeting: en, hi, or.")
    ] = None,
    created_from: Annotated[datetime | None, Query(description="ISO 8601, inclusive.")] = None,
    created_to: Annotated[datetime | None, Query(description="ISO 8601, inclusive.")] = None,
    min_speakers: Annotated[int | None, Query(ge=0)] = None,
    max_speakers: Annotated[int | None, Query(ge=0)] = None,
    sort: Annotated[SortField, Query()] = "created_at",
    order: Annotated[Literal["asc", "desc"], Query()] = "desc",
    cursor: Annotated[str | None, Query(description="next_cursor from the previous page.")] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
) -> MeetingList:
    """Keyset (cursor) pagination: stable while new meetings arrive, fast on large tables."""
    page = await service.page(
        MeetingQuery(
            status=status_,
            language=language.lower() if language else None,
            created_from=created_from,
            created_to=created_to,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            sort=sort,
            order=order,
            cursor=cursor,
            limit=limit,
        )
    )
    return MeetingList(
        items=[await service.to_read(m) for m in page.items],
        total=page.total,
        limit=limit,
        next_cursor=page.next_cursor,
    )


@router.get(
    "/{meeting_id}", response_model=MeetingRead, summary="Get a meeting", responses=_NOT_FOUND
)
async def get_meeting(meeting_id: UUID, service: MeetingServiceDep) -> MeetingRead:
    return await service.to_read(await service.get(meeting_id))


@router.delete(
    "/{meeting_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    summary="Delete a meeting and its stored file",
    responses=_NOT_FOUND,
)
async def delete_meeting(meeting_id: UUID, service: MeetingServiceDep) -> Response:
    await service.delete(meeting_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/{meeting_id}/process",
    response_model=ProcessResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Enqueue the processing pipeline (or resume from a stage)",
    responses={
        **_NOT_FOUND,
        **error_example(
            409,
            "meeting_state_conflict",
            "Meeting is already completed. Use ?force=true to reprocess.",
            "Meeting is already processing or completed",
        ),
    },
)
async def process_meeting(
    meeting_id: UUID,
    request: Request,
    service: MeetingServiceDep,
    pipeline: PipelineDep,
    settings: SettingsDep,
    background_tasks: BackgroundTasks,
    force: Annotated[bool, Query(description="Reprocess even if completed.")] = False,
    from_stage: Annotated[
        str | None,
        Query(description="Reuse the latest successful run's stages before this one."),
    ] = None,
    callback_url: Annotated[
        str | None, Query(description="POSTed with the final status (webhook).")
    ] = None,
    idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
) -> ProcessResponse:
    """Enqueues onto the job queue (``PIPELINE_EXECUTION=queue``) or runs inline for tests.

    Progress: ``GET /meetings/{id}/status`` (snapshot) or ``/status/stream`` (SSE).
    """
    if callback_url:
        validate_callback_url(callback_url, allow_private=settings.webhook_allow_private_hosts)
    meeting = await service.request_processing(meeting_id, force=force)
    request_id = getattr(request.state, "request_id", None)
    if settings.pipeline_execution == "inline":
        background_tasks.add_task(
            pipeline.run, meeting.id, from_stage=from_stage, cancel_check=None
        )
    else:
        queue = build_queue(settings)
        await queue.enqueue_meeting(
            meeting.id,
            from_stage=from_stage,
            idempotency_key=idempotency_key,
            request_id=request_id,
            callback_url=callback_url,
        )
        background_tasks.add_task(queue.close)
    return ProcessResponse(meeting_id=meeting.id, status=meeting.status)


@router.post(
    "/{meeting_id}/cancel",
    response_model=ProcessResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Ask a running pipeline to stop after the current stage",
    responses={**_NOT_FOUND},
)
async def cancel_meeting(
    meeting_id: UUID,
    service: MeetingServiceDep,
    settings: SettingsDep,
) -> ProcessResponse:
    """Sets a cancel flag in Redis. The worker checks it between stages and rolls the
    run to ``cancelled``. In inline mode (no queue) this is a no-op that returns 202."""
    meeting = await service.get(meeting_id)
    if settings.pipeline_execution == "queue" and settings.redis_url:
        queue = build_queue(settings)
        try:
            await queue.cancel(meeting_id)
        finally:
            await queue.close()
    return ProcessResponse(meeting_id=meeting.id, status=meeting.status)


@router.get(
    "/{meeting_id}/speakers",
    response_model=SpeakersResponse,
    summary="Speaker turns from diarization",
    responses={
        **_NOT_FOUND,
        **error_example(
            409,
            "diarization_not_available",
            "Speaker diarization is not available yet.",
            "The meeting has not been diarized yet",
        ),
    },
)
async def get_speakers(meeting_id: UUID, service: MeetingServiceDep) -> SpeakersResponse:
    """Turns sorted by start, labelled Person 1..N by order of first appearance."""
    result = await service.get_diarization(meeting_id)
    return SpeakersResponse(
        meeting_id=meeting_id,
        num_speakers=result.num_speakers,
        speakers=[f"Person {i}" for i in range(1, result.num_speakers + 1)],
        turns=result.turns,
        overlap_regions=result.overlap_regions,
        model_name=result.model_name,
    )


@router.get(
    "/{meeting_id}/transcript",
    response_model=SpeakerTranscriptResponse | TranscriptResponse,
    summary="Speaker-attributed transcript (json, txt, srt, vtt, md); raw ASR with ?view=raw",
    responses={
        200: {
            "content": {
                "text/plain": {
                    "example": "[00:00:00 - 00:00:07] Person 1: आज की मीटिंग का एजेंडा बजट है।\n"
                },
                "application/x-subrip": {
                    "example": "1\n00:00:00,520 --> 00:00:07,100\nPerson 1: आज की मीटिंग ...\n"
                },
                "text/vtt": {
                    "example": "WEBVTT\n\n00:00:00.520 --> 00:00:07.100\n<v Ravi>आज की मीटिंग ...\n"
                },
                "text/markdown": {"example": "# Transcript\n\n**Ravi** · 00:00:00\n\n..."},
            }
        },
        **_NOT_FOUND,
        **error_example(
            409,
            "transcript_not_available",
            "The transcript is not available yet.",
            "The meeting has not been transcribed yet",
        ),
    },
)
async def get_transcript(
    meeting_id: UUID,
    service: MeetingServiceDep,
    format: Annotated[
        Literal["json", "txt", "srt", "vtt", "md"], Query(description="Response format.")
    ] = "json",
    view: Annotated[
        Literal["speaker", "raw"],
        Query(description="speaker: utterances per speaker (default); raw: ASR segments."),
    ] = "speaker",
) -> SpeakerTranscriptResponse | TranscriptResponse | Response:
    """Chronological, speaker-attributed transcript in native script (NFC).

    Display names from ``PATCH /speakers`` are applied in every format.
    """
    if view == "speaker":
        return await _speaker_transcript(meeting_id, service, format)
    if format in ("vtt", "md"):
        raise ValidationError(
            f"format={format} is only available for view=speaker.",
            details={"view": view, "format": format},
        )
    result = await service.get_transcript(meeting_id)
    if format == "txt":
        return PlainTextResponse(to_text(result.segments), media_type="text/plain; charset=utf-8")
    if format == "srt":
        return Response(
            to_srt(result.segments),
            media_type="application/x-subrip; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{meeting_id}.srt"'},
        )
    return TranscriptResponse(meeting_id=meeting_id, **result.model_dump())


@router.get(
    "/{meeting_id}/languages",
    response_model=LanguageSummaryResponse,
    summary="Spoken languages and code-switching",
    responses={
        **_NOT_FOUND,
        **error_example(
            409,
            "language_summary_not_available",
            "Language information is not available.",
            "Language identification has not run for this meeting",
        ),
    },
)
async def get_languages(meeting_id: UUID, service: MeetingServiceDep) -> LanguageSummaryResponse:
    """Per-language and per-speaker time, switch points and code-mixed segment count."""
    summary = await service.get_language_summary(meeting_id)
    return LanguageSummaryResponse(meeting_id=meeting_id, **summary.model_dump())


_MEDIA_TYPES = {
    "txt": "text/plain; charset=utf-8",
    "srt": "application/x-subrip; charset=utf-8",
    "vtt": "text/vtt; charset=utf-8",
    "md": "text/markdown; charset=utf-8",
}


async def _speaker_transcript(
    meeting_id: UUID, service: MeetingServiceDep, format: str
) -> SpeakerTranscriptResponse | Response:
    transcript, names = await service.get_speaker_transcript(meeting_id)
    if format == "json":
        return SpeakerTranscriptResponse(
            meeting_id=meeting_id, speaker_names=names, **transcript.model_dump()
        )
    renderers = {
        "txt": utterances_to_text,
        "srt": utterances_to_srt,
        "vtt": utterances_to_vtt,
        "md": utterances_to_markdown,
    }
    headers = (
        {"Content-Disposition": f'attachment; filename="{meeting_id}.{format}"'}
        if format in ("srt", "vtt")
        else None
    )
    return Response(
        renderers[format](transcript.utterances), media_type=_MEDIA_TYPES[format], headers=headers
    )


@router.patch(
    "/{meeting_id}/speakers",
    response_model=SpeakerNamesResponse,
    summary="Set display names for speakers",
    responses={
        **_NOT_FOUND,
        **error_example(422, "validation_error", "Unknown speaker label(s).", "Unknown label"),
    },
)
async def rename_speakers(
    meeting_id: UUID, body: SpeakerRenameRequest, service: MeetingServiceDep
) -> SpeakerNamesResponse:
    """Map "Person N" labels to names. Labels themselves never change; ``null`` clears a name."""
    names = await service.rename_speakers(meeting_id, body.names)
    return SpeakerNamesResponse(meeting_id=meeting_id, speaker_names=names)


_ANALYTICS_NOT_AVAILABLE = error_example(
    409,
    "analytics_not_available",
    "Analytics are not available yet.",
    "The meeting has not been processed yet",
)


@router.get(
    "/{meeting_id}/analytics",
    response_model=ConversationAnalyticsResponse,
    summary="Speaker-wise conversation statistics and meeting analytics (json, csv)",
    responses={
        200: {
            "content": {
                "text/csv": {
                    "example": "speaker,display_name,speaking_time_seconds,...\n"
                    "Person 1,Ravi,812.3,...\n"
                }
            }
        },
        **_NOT_FOUND,
        **_ANALYTICS_NOT_AVAILABLE,
    },
)
async def get_analytics(
    meeting_id: UUID,
    service: MeetingServiceDep,
    format: Annotated[
        Literal["json", "csv"], Query(description="csv: the per-speaker table.")
    ] = "json",
) -> ConversationAnalyticsResponse | Response:
    """Speaking time, turns, interruptions, WPM, languages and more, per speaker and overall.

    Speaking time comes from diarization turns; overlapping speech counts for every
    overlapping speaker and is reported separately in ``meeting_stats``.
    """
    analytics, names = await service.get_analytics(meeting_id)
    if format == "csv":
        return Response(
            speakers_to_csv(analytics),
            media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{meeting_id}-speakers.csv"'},
        )
    return ConversationAnalyticsResponse(
        meeting_id=meeting_id, speaker_names=names, **analytics.model_dump()
    )


@router.get(
    "/{meeting_id}/analytics/charts/{chart_name}",
    response_class=Response,
    summary="Analytics chart as PNG (speaking-time, timeline)",
    responses={
        200: {"content": {"image/png": {}}},
        **_NOT_FOUND,
        **_ANALYTICS_NOT_AVAILABLE,
        **error_example(
            503,
            "charts_unavailable",
            "Charts need the optional 'viz' extra: uv sync --extra viz.",
            "matplotlib is not installed",
        ),
    },
)
async def get_analytics_chart(
    meeting_id: UUID, chart_name: ChartName, results: ResultServiceDep
) -> Response:
    """Rendered with matplotlib (optional ``viz`` extra), cached in the artifact store."""
    return Response(await results.chart(meeting_id, chart_name), media_type="image/png")


_SUMMARY_NOT_AVAILABLE = error_example(
    409,
    "summary_not_available",
    "The summary is not available yet.",
    "Not processed yet, or summarization failed (see details.summary_error)",
)


@router.get(
    "/{meeting_id}/summary",
    response_model=MeetingSummaryResponse,
    summary="Minutes of meeting: summary, decisions, action items (json, md)",
    responses={
        200: {"content": {"text/markdown": {"example": "# Budget review\n\n## Decisions\n..."}}},
        **_NOT_FOUND,
        **_SUMMARY_NOT_AVAILABLE,
    },
)
async def get_summary(
    meeting_id: UUID,
    service: MeetingServiceDep,
    format: Annotated[Literal["json", "md"], Query(description="Response format.")] = "json",
) -> MeetingSummaryResponse | Response:
    """Every item cites transcript evidence that passed verification.

    JSON uses speaker labels plus ``speaker_names``; Markdown shows display names.
    """
    summary, names = await service.get_summary(meeting_id)
    if format == "md":
        return Response(
            summary_to_markdown(summary, names), media_type="text/markdown; charset=utf-8"
        )
    return MeetingSummaryResponse(
        meeting_id=meeting_id, speaker_names=names, **summary.model_dump()
    )


@router.post(
    "/{meeting_id}/summary/regenerate",
    response_model=RegenerateSummaryResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Regenerate the summary (optionally another language or model)",
    responses={
        **_NOT_FOUND,
        **error_example(
            409,
            "transcript_not_available",
            "The speaker-attributed transcript is not available yet.",
            "Not processed yet, or still processing",
        ),
    },
)
async def regenerate_summary(
    meeting_id: UUID,
    service: MeetingServiceDep,
    regenerator: SummaryRegeneratorDep,
    background_tasks: BackgroundTasks,
    body: RegenerateSummaryRequest | None = None,
) -> RegenerateSummaryResponse:
    """Re-runs only summarization in the background. Poll ``GET /summary``."""
    await service.check_summary_regeneration(meeting_id)
    body = body or RegenerateSummaryRequest()
    background_tasks.add_task(
        regenerator.run, meeting_id, output_language=body.output_language, model=body.model
    )
    return RegenerateSummaryResponse(meeting_id=meeting_id)


_RUN_NOT_FOUND = error_example(404, "run_not_found", "Run ... not found.", "Unknown meeting or run")
RunIdQuery = Annotated[
    UUID | None, Query(description="A run from /runs; default: the latest successful run.")
]


def parse_include(value: str | None) -> list[Section]:
    if value is None:
        return list(SECTIONS)
    parts = [p.strip() for p in value.split(",") if p.strip()]
    invalid = sorted(set(parts) - set(SECTIONS))
    if invalid:
        raise ValidationError(
            "Unknown include section.", details={"invalid": invalid, "allowed": list(SECTIONS)}
        )
    return [p for p in SECTIONS if p in parts]


@router.get(
    "/{meeting_id}/result",
    response_model=MeetingResult,
    response_model_exclude_none=False,
    summary="Consolidated minutes-of-meeting result (schema: /api/v1/schema/meeting-result)",
    responses={**_NOT_FOUND, **_RUN_NOT_FOUND},
)
async def get_result(
    meeting_id: UUID,
    results: ResultServiceDep,
    run_id: RunIdQuery = None,
    include: Annotated[
        str | None,
        Query(description="Comma-separated: transcript, analytics, summary. Default: all."),
    ] = None,
) -> MeetingResult:
    """Metadata, audio quality, languages, speakers (with display names), transcript,
    analytics, summary, verification report, warnings and processing info in one document."""
    return await results.result(meeting_id, run_id, parse_include(include))


@router.get(
    "/{meeting_id}/export",
    response_class=Response,
    summary="Minutes of meeting as docx, pdf, md or json",
    responses={
        200: {"content": {t.split(";")[0]: {} for t in MEDIA_TYPES.values()}},
        **_NOT_FOUND,
        **_RUN_NOT_FOUND,
    },
)
async def export_meeting(
    meeting_id: UUID,
    results: ResultServiceDep,
    format: Annotated[ExportFormat, Query(description="docx, pdf, md or json.")] = "pdf",
    run_id: RunIdQuery = None,
) -> Response:
    """Formal MoM document; cached per run and format until names or the summary change."""
    data, used_run = await results.export(meeting_id, format, run_id)
    stem = f"{meeting_id}-{used_run}" if used_run else str(meeting_id)
    return Response(
        data,
        media_type=MEDIA_TYPES[format],
        headers={"Content-Disposition": f'attachment; filename="minutes-{stem}.{format}"'},
    )


@router.get(
    "/{meeting_id}/utterances",
    response_model=UtterancePage,
    summary="Query utterances by speaker, language, time range and text",
    responses={**_NOT_FOUND, **_RUN_NOT_FOUND},
)
async def list_utterances(
    meeting_id: UUID,
    results: ResultServiceDep,
    speaker: Annotated[str | None, Query(description='Label, e.g. "Person 2".')] = None,
    language: Annotated[str | None, Query(description="Primary language: en, hi, or.")] = None,
    start: Annotated[float | None, Query(ge=0, description="Seconds; overlapping from.")] = None,
    end: Annotated[float | None, Query(ge=0, description="Seconds; overlapping until.")] = None,
    q: Annotated[
        str | None, Query(max_length=200, description="Case-insensitive substring.")
    ] = None,
    run_id: RunIdQuery = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> UtterancePage:
    query = UtteranceQuery(
        speaker=speaker,
        language=language,
        start_from=start,
        end_to=end,
        q=q,
        limit=limit,
        offset=offset,
    )
    page = await results.utterances(meeting_id, query, run_id)
    if page is None:
        raise TranscriptNotAvailableError(
            "The meeting has not been processed yet.", details={"meeting_id": str(meeting_id)}
        )
    return page


@router.get(
    "/{meeting_id}/runs",
    response_model=RunList,
    summary="Processing history, newest first",
    responses=_NOT_FOUND,
)
async def list_runs(meeting_id: UUID, results: ResultServiceDep) -> RunList:
    return await results.runs(meeting_id)


@router.get(
    "/{meeting_id}/runs/{run_id}",
    response_model=MeetingResult,
    summary="The consolidated result of one specific run",
    responses={**_NOT_FOUND, **_RUN_NOT_FOUND},
)
async def get_run(meeting_id: UUID, run_id: UUID, results: ResultServiceDep) -> MeetingResult:
    return await results.result(meeting_id, run_id)
