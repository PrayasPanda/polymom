"""Meeting endpoints: upload, fetch, list, delete."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, File, Form, Query, Response, UploadFile, status

from app.api.deps import MeetingServiceDep
from app.core.exceptions import ValidationError
from app.models.meeting import Meeting
from app.schemas.error import error_example
from app.schemas.meeting import (
    SUPPORTED_LANGUAGES,
    AudioMetadata,
    MeetingCreateResponse,
    MeetingList,
    MeetingRead,
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
