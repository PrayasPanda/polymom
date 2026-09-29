"""The consolidated Minutes-of-Meeting result: one document with everything a client needs.

Published as JSON Schema at ``GET /api/v1/schema/meeting-result``. Bump
``SCHEMA_VERSION`` (semver) on changes: minor for additions, major for breaking ones.
"""

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field

from app.schemas.analytics import ConversationAnalytics, SpeakerStats
from app.schemas.audio import AudioQuality
from app.schemas.language import LanguageSummary
from app.schemas.meeting import MeetingRead
from app.schemas.summary import MeetingSummary, VerificationReport
from app.schemas.transcript import SpeakerTranscript

SCHEMA_VERSION = "1.0.0"
Section = Literal["transcript", "analytics", "summary"]
SECTIONS: tuple[Section, ...] = ("transcript", "analytics", "summary")


class SpeakerInfo(BaseModel):
    label: str = Field(description='Stable diarization label, e.g. "Person 1".')
    display_name: str | None = None
    stats: SpeakerStats | None = Field(default=None, description="From the latest analytics.")


class StageInfo(BaseModel):
    name: str
    status: Literal["completed", "failed", "skipped"] | str
    duration_ms: int | None
    error: str | None = None


class ProcessingInfo(BaseModel):
    run_id: UUID
    status: str
    started_at: datetime
    finished_at: datetime | None
    model_versions: dict[str, str]
    timings_ms: dict[str, int]
    stages: list[StageInfo]
    error: str | None = None


class MeetingResult(BaseModel):
    schema_version: str = SCHEMA_VERSION
    meeting: MeetingRead
    processing: ProcessingInfo | None = Field(description="Null if the meeting never ran.")
    audio_quality: AudioQuality | None = None
    languages: LanguageSummary | None = None
    speakers: list[SpeakerInfo] = Field(default_factory=list)
    transcript: SpeakerTranscript | None = Field(
        default=None, description="Omitted unless included (see ?include=)."
    )
    analytics: ConversationAnalytics | None = None
    summary: MeetingSummary | None = None
    verification_report: VerificationReport | None = None
    warnings: list[str] = Field(
        default_factory=list,
        description="Audio quality, alignment and processing warnings, human-readable.",
    )
    included: list[Section] = Field(description="Which optional sections were requested.")


class UtterancePage(BaseModel):
    meeting_id: UUID
    run_id: UUID
    items: list["UtteranceRow"]
    total: int
    limit: int
    offset: int


class UtteranceRow(BaseModel):
    id: int
    speaker: str
    speaker_name: str | None
    start: float
    end: float
    text: str
    primary_language: str | None
    is_code_mixed: bool
    has_overlap: bool
    alignment_precision: str


class RunSummary(BaseModel):
    run_id: UUID
    status: str
    started_at: datetime
    finished_at: datetime | None
    model_versions: dict[str, str]
    error: str | None
    is_default: bool = Field(description="The run served when no run_id is given.")


class RunList(BaseModel):
    meeting_id: UUID
    items: list[RunSummary]


class SearchHit(BaseModel):
    kind: Literal["utterance", "summary"]
    meeting_id: UUID
    meeting_title: str | None
    run_id: UUID
    utterance_id: int | None
    speaker: str | None
    speaker_name: str | None
    start: float | None
    text: str


class SearchResults(BaseModel):
    query: str
    items: list[SearchHit]


UtterancePage.model_rebuild()
