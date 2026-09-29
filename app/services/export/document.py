"""Format-neutral view of the minutes, built once from a :class:`MeetingResult`.

Every renderer (md, docx, pdf) lays out the same sections in the same order, and
display names are applied here, so the formats never disagree.
"""

from dataclasses import dataclass, field

from app.schemas.result import MeetingResult
from app.services.summarization.chunker import format_timestamp

LANGUAGE_NAMES = {"en": "English", "hi": "Hindi", "or": "Odia"}


@dataclass
class Participant:
    name: str
    label: str
    speaking_seconds: float | None
    share_percent: float | None


@dataclass
class TranscriptLine:
    start: str
    end: str
    speaker: str
    text: str


@dataclass
class MinutesDocument:
    title: str
    date: str
    duration: str
    participants: list[Participant]
    languages: list[str]
    executive_summary: str | None
    agenda: list[str] = field(default_factory=list)
    key_points: list[tuple[str, str]] = field(default_factory=list)
    decisions: list[list[str]] = field(default_factory=list)
    action_items: list[list[str]] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    transcript: list[TranscriptLine] = field(default_factory=list)
    footer: str = ""

    DECISION_HEADERS = ("#", "Decision", "By", "Confidence", "Status")
    ACTION_HEADERS = ("#", "Task", "Owner", "Due", "Priority")


def build_document(result: MeetingResult) -> MinutesDocument:
    names = {s.label: s.display_name for s in result.speakers if s.display_name}

    def who(label: str) -> str:
        return names.get(label, label)

    meeting = result.meeting
    summary = result.summary
    stats = {s.label: s.stats for s in result.speakers}
    participants = [
        Participant(
            name=who(s.label),
            label=s.label,
            speaking_seconds=st.speaking_time_seconds if (st := stats.get(s.label)) else None,
            share_percent=st.speaking_time_percent_of_speech if st else None,
        )
        for s in result.speakers
        if s.label != "Unknown"
    ]
    participants.sort(key=lambda p: -(p.speaking_seconds or 0))
    languages = sorted(
        {
            lang
            for u in (result.transcript.utterances if result.transcript else [])
            for lang in u.languages_present
        }
        | set(meeting.detected_languages)
    )
    duration = meeting.duration_seconds or (
        result.analytics.meeting_stats.meeting_duration_seconds if result.analytics else 0.0
    )
    doc = MinutesDocument(
        title=(summary.title if summary else None) or meeting.title or meeting.original_filename,
        date=f"{meeting.created_at:%Y-%m-%d %H:%M} UTC",
        duration=format_timestamp(duration or 0.0),
        participants=participants,
        languages=[LANGUAGE_NAMES.get(x, x) for x in languages],
        executive_summary=summary.executive_summary if summary else None,
    )
    if summary:
        doc.agenda = list(summary.agenda_topics)
        doc.key_points = [(k.title, k.description) for k in summary.key_points]
        doc.decisions = [
            [
                str(i),
                d.decision,
                who(d.made_by),
                d.confidence,
                d.status + (f": {d.note}" if d.note else ""),
            ]
            for i, d in enumerate(summary.decisions, 1)
        ]
        doc.action_items = [
            [
                str(i),
                a.task,
                who(a.owner),
                (a.due_date.raw + (f" ({a.due_date.iso})" if a.due_date.iso else ""))
                if a.due_date
                else "-",
                a.priority,
            ]
            for i, a in enumerate(summary.action_items, 1)
        ]
        doc.open_questions = [f"{q.question} ({who(q.raised_by)})" for q in summary.open_questions]
        r = summary.verification_report
        doc.footer = (
            f"Generated with {summary.model_info.provider}/{summary.model_info.model}. "
            f"{r.checked} items verified against the transcript: {r.passed} passed, "
            f"{r.downgraded} downgraded, {r.dropped} dropped."
        )
    if result.transcript:
        doc.transcript = [
            TranscriptLine(
                format_timestamp(u.start), format_timestamp(u.end), who(u.speaker), u.text
            )
            for u in result.transcript.utterances
        ]
    return doc


def participant_line(p: Participant) -> str:
    if p.speaking_seconds is None:
        return p.name
    share = f", {p.share_percent:.0f}%" if p.share_percent is not None else ""
    return f"{p.name}: {format_timestamp(p.speaking_seconds)}{share}"
