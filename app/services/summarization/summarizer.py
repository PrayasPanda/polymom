"""Grounded minutes of meeting from the speaker-attributed transcript.

Short meetings (<= ``SUMMARY_SINGLE_PASS_TOKENS``) take one LLM call. Longer ones
use map-reduce: extract items per chunk, merge them deterministically
(:mod:`.reducer`), then one call writes the title and executive summary. Every
item is then checked against the transcript (:mod:`.verifier`).
"""

from datetime import UTC, date, datetime
from typing import TypeVar

from pydantic import BaseModel

from app.core.config import Settings
from app.core.logging import get_logger
from app.schemas.analytics import ConversationAnalytics
from app.schemas.summary import (
    ChunkExtraction,
    LLMUsageTotals,
    MeetingSummary,
    ModelInfo,
    SummaryDraft,
    SummaryHeader,
    VerificationReport,
)
from app.schemas.transcript import SpeakerTranscript
from app.services.llm.base import LLMClient
from app.services.llm.tracing import NoopTracer, Tracer
from app.services.summarization import prompts
from app.services.summarization.chunker import (
    chunk_utterances,
    estimate_tokens,
    format_transcript,
)
from app.services.summarization.injection import detect_injections
from app.services.summarization.reducer import merge_extractions
from app.services.summarization.verifier import Verifier

logger = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)

LANGUAGE_NAMES = {
    "en": "English",
    "hi": "Hindi (Devanagari script)",
    "or": "Odia (Odia script)",
}


def analytics_context(analytics: ConversationAnalytics | None) -> str:
    """Compact speaker statistics for the prompt (context, never evidence)."""
    if analytics is None:
        return "not available"
    m = analytics.meeting_stats
    lines = [
        f"duration {m.meeting_duration_seconds:.0f}s, {m.num_speakers} speakers, "
        f"balance: {m.participation_balance}, dominant speaker: {m.dominant_speaker or '-'}"
    ]
    lines += [
        f"{s.speaker}: {s.speaking_time_seconds:.0f}s "
        f"({s.speaking_time_percent_of_speech:.0f}% of speech), {s.word_count} words, "
        f"{s.questions_asked} questions"
        for s in analytics.speakers
    ]
    return "\n".join(lines)


class Summarizer:
    def __init__(self, llm: LLMClient, settings: Settings, tracer: Tracer | None = None) -> None:
        self.llm = llm
        self.settings = settings
        self.tracer = tracer or NoopTracer()

    async def _call(
        self,
        template: str,
        schema: type[T],
        **context: object,
    ) -> T:
        system = prompts.render("system.j2")
        user = prompts.render(template, **context)
        version = f"{prompts.prompt_version('system.j2')}+{prompts.prompt_version(template)}"
        try:
            return await self.llm.generate_structured(
                system, user, schema, name=template.removesuffix(".j2")
            )
        finally:
            if self.llm.calls:
                self.tracer.record(self.llm.calls[-1], prompt_version=version, input=user)

    async def summarize(
        self,
        transcript: SpeakerTranscript,
        analytics: ConversationAnalytics | None = None,
        *,
        output_language: str | None = None,
        meeting_date: date | None = None,
    ) -> MeetingSummary:
        language = output_language or self.settings.summary_output_language
        utterances = transcript.utterances
        flags = detect_injections(utterances)
        if flags:
            logger.warning("prompt_injection_suspected", count=len(flags))
        common = {
            "meeting_date": (meeting_date or datetime.now(UTC).date()).isoformat(),
            "output_language": LANGUAGE_NAMES.get(language, language),
        }
        calls_before = len(self.llm.calls)
        text = format_transcript(utterances)
        chunks = 0
        if not utterances:
            strategy = "single_pass"
            header = SummaryHeader(
                title="No speech detected",
                executive_summary="The recording contains no transcribed speech.",
            )
            items = ChunkExtraction()
        elif estimate_tokens(text) <= self.settings.summary_single_pass_tokens:
            strategy, chunks = "single_pass", 1
            draft = await self._call(
                "single_pass.j2",
                SummaryDraft,
                transcript=text,
                analytics=analytics_context(analytics),
                **common,
            )
            header, items = draft, draft
        else:
            strategy = "map_reduce"
            parts = chunk_utterances(
                utterances,
                self.settings.summary_chunk_tokens,
                self.settings.summary_chunk_overlap_utterances,
            )
            chunks = len(parts)
            extractions = []
            # ponytail: sequential map calls; asyncio.gather with a semaphore if latency matters.
            for i, part in enumerate(parts, start=1):
                extraction = await self._call(
                    "map.j2",
                    ChunkExtraction,
                    transcript=format_transcript(part),
                    part=i,
                    parts=len(parts),
                    **common,
                )
                extractions.append(extraction)
            items = merge_extractions(extractions)
            reduced = await self._call(
                "reduce.j2",
                SummaryHeader,
                items=items.model_dump_json(indent=1),
                analytics=analytics_context(analytics),
                **common,
            )
            header = reduced
        self.tracer.flush()

        verified, report = Verifier(utterances, self.settings.evidence_match_threshold).verify(
            ChunkExtraction.model_validate(items.model_dump()), flags
        )
        return self._build(
            header,
            verified,
            report,
            language,
            utterances,
            strategy,
            chunks,
            calls_before,
            transcript,
        )

    def _build(
        self,
        header: SummaryHeader,
        items: ChunkExtraction,
        report: VerificationReport,
        language: str,
        utterances: object,
        strategy: str,
        chunks: int,
        calls_before: int,
        transcript: SpeakerTranscript,
    ) -> MeetingSummary:
        calls = self.llm.calls[calls_before:]
        templates = ["system.j2"] + (
            ["single_pass.j2"] if strategy == "single_pass" else ["map.j2", "reduce.j2"]
        )
        return MeetingSummary(
            title=header.title,
            executive_summary=header.executive_summary,
            agenda_topics=header.agenda_topics,
            key_points=items.key_points,
            decisions=items.decisions,
            action_items=items.action_items,
            open_questions=items.open_questions,
            overall_sentiment=header.overall_sentiment,
            output_language=language,
            source_languages=sorted(
                {lang for u in transcript.utterances for lang in u.languages_present}
            ),
            model_info=ModelInfo(
                provider=self.llm.provider,
                model=self.llm.model,
                prompt_version=",".join(prompts.prompt_version(t) for t in templates),
                strategy="map_reduce" if strategy == "map_reduce" else "single_pass",
                num_chunks=chunks,
                temperature=self.settings.llm_temperature,
                usage=LLMUsageTotals(
                    calls=len(calls),
                    prompt_tokens=sum(c.prompt_tokens for c in calls),
                    completion_tokens=sum(c.completion_tokens for c in calls),
                    cost_usd=round(sum(c.cost_usd for c in calls), 6),
                    latency_ms=sum(c.latency_ms for c in calls),
                    repair_attempts=sum(c.attempts - 1 for c in calls),
                ),
            ),
            generated_at=datetime.now(UTC),
            verification_report=report,
        )
