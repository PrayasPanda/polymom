"""Meeting endpoints: upload, fetch, list, delete, process."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, File, Form, Query, Response, UploadFile, status

from app.api.deps import MeetingServiceDep, PipelineDep
from app.core.exceptions import ValidationError
from app.models.meeting import Meeting
from app.schemas.audio import AudioQuality
from app.schemas.diarization import SpeakersResponse
from app.schemas.error import error_example
from app.schemas.meeting import (
    SUPPORTED_LANGUAGES,
    AudioMetadata,
    MeetingCreateResponse,
    MeetingList,
    MeetingRead,
    ProcessResponse,
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
