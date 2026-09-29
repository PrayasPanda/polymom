"""Meeting-level analytics and the full :class:`ConversationAnalytics` report. Pure functions."""

import math
from collections.abc import Sequence
from itertools import pairwise

from app.schemas.analytics import (
    ConversationAnalytics,
    MeetingStats,
    Monologue,
    ParticipationBalance,
    SpeakerStats,
    TimelineBucket,
)
from app.schemas.diarization import SpeakerTurn
from app.schemas.language import LanguageSummary
from app.schemas.transcript import UNKNOWN_SPEAKER, SpeakerTranscript, Utterance
from app.services.analytics.speaker_stats import (
    Interval,
    compute_speaker_stats,
    floor_runs,
    intersect,
    language_shares,
    merge_intervals,
    overlap_intervals,
    speaker_intervals,
    total_length,
    utterance_language_seconds,
)


def gini(values: Sequence[float]) -> float:
    """Gini coefficient: 0 when all values are equal, (n-1)/n when one holds everything."""
    n, total = len(values), sum(values)
    if n == 0 or total <= 0:
        return 0.0
    diffs = sum(abs(a - b) for a in values for b in values)
    return diffs / (2 * n * total)


def participation_balance(
    gini_value: float, num_speakers: int, balanced_max: float, dominated_min: float
) -> ParticipationBalance:
    if num_speakers < 2:
        return "not applicable"
    if gini_value <= balanced_max:
        return "balanced"
    if gini_value >= dominated_min:
        return "dominated"
    return "moderately dominated"


def timeline(
    intervals: dict[str, list[Interval]], duration: float, bucket_seconds: float
) -> list[TimelineBucket]:
    """Speaking seconds per speaker in consecutive ``bucket_seconds`` windows."""
    buckets: list[TimelineBucket] = []
    for i in range(math.ceil(duration / bucket_seconds) if duration > 0 else 0):
        window = [(i * bucket_seconds, min((i + 1) * bucket_seconds, duration))]
        buckets.append(
            TimelineBucket(
                start=window[0][0],
                end=window[0][1],
                speakers={
                    speaker: total_length(intersect(spans, window))
                    for speaker, spans in sorted(intervals.items())
                },
            )
        )
    return buckets


def utterance_language_switches(utterances: Sequence[Utterance]) -> int:
    langs = [u.primary_language for u in utterances if u.primary_language]
    return sum(a != b for a, b in pairwise(langs))


def compute_meeting_stats(
    turns: Sequence[SpeakerTurn],
    utterances: Sequence[Utterance],
    speakers: Sequence[SpeakerStats],
    *,
    meeting_duration: float,
    gini_balanced_max: float,
    gini_dominated_min: float,
    language_summary: LanguageSummary | None = None,
) -> MeetingStats:
    intervals = speaker_intervals(turns)
    speech = total_length(merge_intervals(s for v in intervals.values() for s in v))
    overlap = total_length(
        merge_intervals(s for v in overlap_intervals(intervals).values() for s in v)
    )
    summed = sum(total_length(v) for v in intervals.values())
    times = [s.speaking_time_seconds for s in speakers]
    g = gini(times)
    runs = floor_runs(turns)
    longest = max(runs, key=lambda r: (r.end - r.start, -r.start), default=None)
    attributed = [u for u in utterances if u.speaker != UNKNOWN_SPEAKER]
    active = speakers and speakers[0].speaking_time_seconds > 0
    return MeetingStats(
        meeting_duration_seconds=meeting_duration,
        total_speech_seconds=speech,
        total_silence_seconds=max(meeting_duration - speech, 0.0),
        silence_percent=(
            max(meeting_duration - speech, 0.0) / meeting_duration * 100
            if meeting_duration > 0
            else 0.0
        ),
        num_speakers=len(speakers),
        total_utterances=len(attributed),
        total_words=sum(len(u.words) for u in attributed),
        total_turn_switches=max(len(runs) - 1, 0),
        overlap_seconds=overlap,
        overlap_percent=overlap / speech * 100 if speech > 0 else 0.0,
        overlap_double_counted_seconds=max(summed - speech, 0.0),
        gini_coefficient=g,
        participation_balance=participation_balance(
            g, len(speakers), gini_balanced_max, gini_dominated_min
        ),
        dominant_speaker=speakers[0].speaker if active else None,
        least_active_speaker=speakers[-1].speaker if active else None,
        language_distribution=(
            language_summary.languages
            if language_summary
            else language_shares(utterance_language_seconds(attributed))
        ),
        num_language_switches=(
            language_summary.num_switches
            if language_summary
            else utterance_language_switches(attributed)
        ),
        longest_monologue=(
            Monologue(
                speaker=longest.speaker,
                start=longest.start,
                end=longest.end,
                duration=longest.end - longest.start,
            )
            if longest
            else None
        ),
    )


def build_analytics(
    turns: Sequence[SpeakerTurn],
    transcript: SpeakerTranscript,
    *,
    language_summary: LanguageSummary | None = None,
    interruption_min_overlap: float,
    bucket_seconds: float,
    gini_balanced_max: float,
    gini_dominated_min: float,
) -> ConversationAnalytics:
    """The full report. Meeting duration is the recording length, extended to cover all speech."""
    duration = max(
        [transcript.total_duration]
        + [t.end for t in turns]
        + [u.end for u in transcript.utterances]
    )
    speakers = compute_speaker_stats(
        turns,
        transcript.utterances,
        meeting_duration=duration,
        interruption_min_overlap=interruption_min_overlap,
        language_summary=language_summary,
    )
    return ConversationAnalytics(
        meeting_stats=compute_meeting_stats(
            turns,
            transcript.utterances,
            speakers,
            meeting_duration=duration,
            gini_balanced_max=gini_balanced_max,
            gini_dominated_min=gini_dominated_min,
            language_summary=language_summary,
        ),
        speakers=speakers,
        timeline=timeline(speaker_intervals(turns), duration, bucket_seconds),
    )
