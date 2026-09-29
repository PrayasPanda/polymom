"""Meeting endpoints: upload, process, speakers, transcript, analytics, summary."""

from typing import Annotated, Literal
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, File, Form, Query, Response, UploadFile, status
from fastapi.responses import PlainTextResponse
from starlette.concurrency import run_in_threadpool

from app.api.deps import MeetingServiceDep, PipelineDep, SummaryRegeneratorDep
from app.core.exceptions import ValidationError
from app.models.meeting import Meeting
from app.schemas.analytics import ConversationAnalyticsResponse
from app.schemas.asr import TranscriptResponse
from app.schemas.audio import AudioQuality
from app.schemas.diarization import SpeakersResponse
from app.schemas.error import error_example
from app.schemas.language import LanguageSummaryResponse
from app.schemas.meeting import (
    SUPPORTED_LANGUAGES,
    AudioMetadata,
    MeetingCreateResponse,
    MeetingList,
    MeetingRead,
    ProcessResponse,
)
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
from app.services.analytics.charts import ChartName, render_chart
from app.services.analytics.export import speakers_to_csv
from app.services.summarization.render import summary_to_markdown
from app.utils.subtitles import (
    to_srt,
    to_text,
    utterances_to_markdown,
    utterances_to_srt,
    utterances_to_text,
    utterances_to_vtt,
)

router = APIRouter(prefix="/meetings", tags=["meetings"])

_NOT_FOUND = error_example(404, "meeting_not_found", "Meeting ... not found.", "Unknown meeting")


def to_read(meeting: Meeting) -> MeetingRead:
    return MeetingRead(
        meeting_id=meeting.id,
        title=meeting.title,
        original_filename=meeting.original_filename,
        mime_type=meeting.mime_type,
        size_bytes=meeting.size_bytes,
        duration_seconds=meeting.duration_seconds,
        audio_metadata=AudioMetadata.model_validate(meeting.audio_metadata),
        audio_quality=(
            AudioQuality.model_validate(meeting.audio_quality) if meeting.audio_quality else None
        ),
        languages_hint=meeting.languages_hint,
        expected_speakers=meeting.expected_speakers,
        status=meeting.status,
        error=meeting.error,
        created_at=meeting.created_at,
        updated_at=meeting.updated_at,
    )


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
) -> MeetingCreateResponse:
    """Validate the upload, store it and queue the meeting for processing."""
    langs = parse_languages(languages)
    try:
        meeting = await service.create(
            source=file,
            filename=file.filename,
            title=title.strip() if title and title.strip() else None,
            expected_speakers=expected_speakers,
            languages=langs,
        )
    finally:
        await file.close()
    return MeetingCreateResponse(
        meeting_id=meeting.id, status=meeting.status, created_at=meeting.created_at
    )


@router.get("", response_model=MeetingList, summary="List meetings, newest first")
async def list_meetings(
    service: MeetingServiceDep,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> MeetingList:
    items, total = await service.list(limit=limit, offset=offset)
    return MeetingList(items=[to_read(m) for m in items], total=total, limit=limit, offset=offset)


@router.get(
    "/{meeting_id}", response_model=MeetingRead, summary="Get a meeting", responses=_NOT_FOUND
)
async def get_meeting(meeting_id: UUID, service: MeetingServiceDep) -> MeetingRead:
    return to_read(await service.get(meeting_id))


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
    summary="Run the processing pipeline for a meeting",
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
    service: MeetingServiceDep,
    pipeline: PipelineDep,
    background_tasks: BackgroundTasks,
    force: Annotated[bool, Query(description="Reprocess even if processing or completed.")] = False,
) -> ProcessResponse:
    """Queue the pipeline in the background. Poll ``GET /meetings/{id}`` for the status.

    Runs in-process via ``BackgroundTasks`` for now; a real job queue replaces it later.
    """
    meeting = await service.request_processing(meeting_id, force=force)
    background_tasks.add_task(pipeline.run, meeting.id)
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
    meeting_id: UUID, chart_name: ChartName, service: MeetingServiceDep
) -> Response:
    """Rendered with matplotlib; needs the optional ``viz`` extra. Uses display names."""
    analytics, names = await service.get_analytics(meeting_id)
    turns = (await service.get_diarization(meeting_id)).turns if chart_name == "timeline" else []
    png = await run_in_threadpool(render_chart, chart_name, analytics, names, turns)
    return Response(png, media_type="image/png")


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
