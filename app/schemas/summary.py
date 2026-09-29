"""Grounded Minutes-of-Meeting schemas.

Every item carries ``evidence`` pointing at transcript utterances. The LLM fills
the ``*Draft`` / ``ChunkExtraction`` shapes; the verifier checks the evidence
against the transcript and produces the final :class:`MeetingSummary`.
Speakers are always diarization labels ("Person 2"); display names are applied
only when rendering.
"""

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field

Confidence = Literal["high", "medium", "low"]
Priority = Literal["high", "medium", "low"]
Sentiment = Literal["positive", "neutral", "negative", "mixed"]
UNASSIGNED = "Unassigned"
GROUP = "group"


class Evidence(BaseModel):
    utterance_ids: list[int] = Field(min_length=1, description="Ids from the [uN] line prefixes.")
    speaker: str = Field(description='Label of the speaker quoted, e.g. "Person 2".')
    quote: str = Field(description="Short verbatim quote in the original script and language.")
    start: float | None = Field(default=None, description="Filled from the transcript.")
    end: float | None = None


class KeyPoint(BaseModel):
    title: str
    description: str
    speakers_involved: list[str] = Field(default_factory=list)
    evidence: list[Evidence] = Field(min_length=1)


class Decision(BaseModel):
    decision: str
    topic: str = Field(default="", description="Short noun phrase; used to detect supersession.")
    made_by: str = Field(description='Speaker label or "group".')
    rationale: str = ""
    evidence: list[Evidence] = Field(min_length=1)
    confidence: Confidence = "medium"
    status: Literal["active", "superseded"] = "active"
    note: str | None = None


class DueDate(BaseModel):
    raw: str = Field(description='As stated, e.g. "kal", "agle hafte", "by EOD Friday".')
    iso: str | None = Field(
        default=None, description="YYYY-MM-DD only if unambiguous from the meeting date."
    )


class ActionItem(BaseModel):
    task: str
    owner: str = Field(description='Speaker label or "Unassigned". Never a guessed name.')
    due_date: DueDate | None = None
    priority: Priority = "medium"
    evidence: list[Evidence] = Field(min_length=1)
    confidence: Confidence = "medium"


class OpenQuestion(BaseModel):
    question: str
    raised_by: str
    evidence: list[Evidence] = Field(min_length=1)


class ChunkExtraction(BaseModel):
    """Map-step output for one transcript chunk."""

    key_points: list[KeyPoint] = Field(default_factory=list)
    decisions: list[Decision] = Field(default_factory=list)
    action_items: list[ActionItem] = Field(default_factory=list)
    open_questions: list[OpenQuestion] = Field(default_factory=list)


class SummaryHeader(BaseModel):
    """Reduce-step output: the narrative parts, written from the merged items."""

    title: str
    executive_summary: str = Field(description="5 to 8 sentences.")
    agenda_topics: list[str] = Field(default_factory=list)
    overall_sentiment: Sentiment = "neutral"


class SummaryDraft(SummaryHeader, ChunkExtraction):
    """Single-pass output: header and items in one call."""


class VerificationIssue(BaseModel):
    item_type: Literal["key_point", "decision", "action_item", "open_question"]
    item: str
    action: Literal["dropped", "downgraded", "evidence_removed", "owner_reset"]
    reason: str


class InjectionFlag(BaseModel):
    utterance_id: int
    speaker: str
    pattern: str
    text: str


class VerificationReport(BaseModel):
    checked: int = Field(description="Items checked.")
    passed: int = Field(description="Items kept without changes.")
    dropped: int
    downgraded: int
    evidence_checked: int
    evidence_passed: int
    issues: list[VerificationIssue] = Field(default_factory=list)
    injection_flags: list[InjectionFlag] = Field(default_factory=list)

    @property
    def grounding_pass_rate(self) -> float:
        return self.evidence_passed / self.evidence_checked if self.evidence_checked else 1.0


class LLMUsageTotals(BaseModel):
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: int = 0
    repair_attempts: int = 0


class ModelInfo(BaseModel):
    provider: str
    model: str
    prompt_version: str
    strategy: Literal["single_pass", "map_reduce"]
    num_chunks: int
    temperature: float
    usage: LLMUsageTotals


class MeetingSummary(BaseModel):
    title: str
    executive_summary: str
    agenda_topics: list[str]
    key_points: list[KeyPoint]
    decisions: list[Decision]
    action_items: list[ActionItem]
    open_questions: list[OpenQuestion]
    overall_sentiment: Sentiment
    output_language: str
    source_languages: list[str]
    model_info: ModelInfo
    generated_at: datetime
    verification_report: VerificationReport


class MeetingSummaryResponse(MeetingSummary):
    meeting_id: UUID
    speaker_names: dict[str, str] = Field(default_factory=dict)


class RegenerateSummaryRequest(BaseModel):
    output_language: Literal["en", "hi", "or"] | None = None
    model: str | None = Field(default=None, max_length=200, description="Overrides LLM_MODEL.")


class RegenerateSummaryResponse(BaseModel):
    meeting_id: UUID
    status: Literal["queued"] = "queued"
