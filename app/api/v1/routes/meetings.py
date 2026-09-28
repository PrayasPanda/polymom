"""Meeting endpoints (stubs)."""

from fastapi import APIRouter, status

from app.api.deps import MeetingRepoDep
from app.core.exceptions import NotFoundError
from app.models.meeting import Meeting
from app.schemas.meeting import MeetingCreateResponse, MeetingRead

router = APIRouter(prefix="/meetings", tags=["meetings"])


def _to_read(meeting: Meeting) -> MeetingRead:
    return MeetingRead(
        meeting_id=meeting.id,
        status=meeting.status,
        filename=meeting.filename,
        created_at=meeting.created_at,
    )


@router.post("", response_model=MeetingCreateResponse, status_code=status.HTTP_202_ACCEPTED)
async def create_meeting() -> MeetingCreateResponse:
    """Upload a recording and enqueue processing.

    TODO(prompt-2): accept multipart upload, validate size/extension, persist, enqueue.
    """
    raise NotImplementedError


@router.get("", response_model=list[MeetingRead])
async def list_meetings(repo: MeetingRepoDep) -> list[MeetingRead]:
    """List meetings."""
    return [_to_read(m) for m in await repo.list_all()]


@router.get("/{meeting_id}", response_model=MeetingRead)
async def get_meeting(meeting_id: str, repo: MeetingRepoDep) -> MeetingRead:
    """Fetch the status of a meeting. TODO: include transcript, stats and summary."""
    meeting = await repo.get(meeting_id)
    if meeting is None:
        raise NotFoundError(f"Meeting {meeting_id!r} not found.")
    return _to_read(meeting)
