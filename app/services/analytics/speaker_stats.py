"""Per-speaker conversation statistics. Pure functions, no I/O.

Sources of truth:

- **Time** (speaking time, turns, overlap, interruptions, first/last spoke) comes
  from diarization turns, so pauses inside a turn count the same way for everyone.
- **Words, segments and questions** come from the aligned utterances.
- **Overlap** counts in full for each overlapping speaker, so per-speaker times
  can add up to more than the total speech time; the meeting-level overlap
  explains the difference.
"""

import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from typing import NamedTuple

from app.schemas.analytics import SpeakerStats, TimeSpan
from app.schemas.diarization import SpeakerTurn
from app.schemas.language import LanguageShare, LanguageSummary
from app.schemas.transcript import UNKNOWN_SPEAKER, Utterance

Interval = tuple[float, float]

# Small, documented list of question words; a heuristic, not a parser.
HINDI_QUESTION_MARKERS = frozenset(
    {"क्या", "क्यों", "कैसे", "कब", "कहाँ", "कहां", "कौन", "कौनसा", "कितना", "कितने", "कितनी"}
)
ODIA_QUESTION_MARKERS = frozenset(
    {"କଣ", "କ'ଣ", "କ’ଣ", "କି", "କାହିଁକି", "କିପରି", "କେମିତି", "କେବେ", "କେଉଁଠି", "କିଏ", "କେତେ"}  # noqa: RUF001
)
QUESTION_MARKERS = HINDI_QUESTION_MARKERS | ODIA_QUESTION_MARKERS
_STRIP = '.,;:!?"()[]“”‘।॥'  # noqa: RUF001 - curly quotes are intentional


class FloorRun(NamedTuple):
    """Consecutive turns by one speaker: one "turn" in the turn-taking sense."""

    speaker: str
    start: float
    end: float


# --- interval helpers ---


def merge_intervals(intervals: Iterable[Interval]) -> list[Interval]:
    """Sorted, non-overlapping union; empty intervals are dropped."""
    merged: list[Interval] = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def total_length(intervals: Iterable[Interval]) -> float:
    return sum(end - start for start, end in intervals)


def intersect(a: Sequence[Interval], b: Sequence[Interval]) -> list[Interval]:
    """Intersection of two merged interval lists."""
    out: list[Interval] = []
    i = j = 0
    while i < len(a) and j < len(b):
        start, end = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if start < end:
            out.append((start, end))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return out


def speaker_intervals(turns: Iterable[SpeakerTurn]) -> dict[str, list[Interval]]:
    """Merged speech intervals per speaker label."""
    raw: dict[str, list[Interval]] = defaultdict(list)
    for turn in turns:
        raw[turn.speaker_label].append((turn.start, turn.end))
    return {speaker: merge_intervals(spans) for speaker, spans in raw.items()}


def overlap_intervals(intervals: dict[str, list[Interval]]) -> dict[str, list[Interval]]:
    """Per speaker, the time they speak while at least one other speaker does."""
    result: dict[str, list[Interval]] = {}
    for speaker, own in intervals.items():
        others = merge_intervals(s for k, v in intervals.items() if k != speaker for s in v)
        result[speaker] = intersect(own, others)
    return result


# --- turn taking ---


def floor_runs(turns: Iterable[SpeakerTurn]) -> list[FloorRun]:
    """Turns in start order with consecutive same-speaker turns merged."""
    runs: list[FloorRun] = []
    for turn in sorted(turns, key=lambda t: (t.start, t.speaker_label)):
        if runs and runs[-1].speaker == turn.speaker_label:
            last = runs[-1]
            runs[-1] = FloorRun(last.speaker, last.start, max(last.end, turn.end))
        else:
            runs.append(FloorRun(turn.speaker_label, turn.start, turn.end))
    return runs


def interruptions(turns: Sequence[SpeakerTurn], min_overlap: float) -> list[tuple[str, str]]:
    """``(interrupter, interrupted)`` pairs.

    A turn interrupts when it starts strictly inside another speaker's turn and
    overlaps it by more than ``min_overlap`` seconds. Each turn interrupts at
    most one speaker: the one it overlaps most.
    """
    # ponytail: O(n^2) over turns; fine for meeting-sized inputs, sort+sweep if it's ever hot.
    pairs: list[tuple[str, str]] = []
    for b in turns:
        if any(
            t is not b and t.speaker_label == b.speaker_label and t.start <= b.start < t.end
            for t in turns
        ):
            continue  # the speaker was already talking: a continuation, not an interruption
        best: tuple[float, str] | None = None
        for a in turns:
            if a.speaker_label == b.speaker_label or not a.start < b.start < a.end:
                continue
            overlap = min(a.end, b.end) - b.start
            if overlap > min_overlap and (best is None or overlap > best[0]):
                best = (overlap, a.speaker_label)
        if best is not None:
            pairs.append((b.speaker_label, best[1]))
    return pairs


# --- text ---


def is_question(text: str) -> bool:
    """Ends in "?" or contains a Hindi/Odia question word."""
    stripped = text.strip()
    if stripped.endswith(("?", "？")):  # noqa: RUF001 - full-width question mark
        return True
    return any(token.strip(_STRIP) in QUESTION_MARKERS for token in stripped.split())


def language_shares(durations: dict[str, float]) -> list[LanguageShare]:
    """Seconds per language as shares, longest first."""
    total = sum(durations.values())
    return [
        LanguageShare(
            language=lang,
            duration_seconds=round(seconds, 2),
            percentage=round(seconds / total * 100, 2) if total else 0.0,
        )
        for lang, seconds in sorted(durations.items(), key=lambda kv: (-kv[1], kv[0]))
    ]


def utterance_language_seconds(utterances: Iterable[Utterance]) -> dict[str, float]:
    seconds: dict[str, float] = defaultdict(float)
    for u in utterances:
        if u.primary_language:
            seconds[u.primary_language] += u.duration
    return dict(seconds)


# --- per speaker ---


def compute_speaker_stats(
    turns: Sequence[SpeakerTurn],
    utterances: Sequence[Utterance],
    *,
    meeting_duration: float,
    interruption_min_overlap: float,
    language_summary: LanguageSummary | None = None,
) -> list[SpeakerStats]:
    """Stats for every known speaker, most speaking time first.

    ``Unknown`` utterances are excluded, so segment shares sum to 100 over
    attributed utterances.
    """
    intervals = speaker_intervals(turns)
    overlaps = overlap_intervals(intervals)
    total_speech = total_length(merge_intervals(s for v in intervals.values() for s in v))
    runs = Counter(run.speaker for run in floor_runs(turns))
    pairs = interruptions(turns, interruption_min_overlap)
    made = Counter(a for a, _ in pairs)
    received = Counter(b for _, b in pairs)

    by_speaker: dict[str, list[Utterance]] = defaultdict(list)
    for u in utterances:
        if u.speaker != UNKNOWN_SPEAKER:
            by_speaker[u.speaker].append(u)
    total_utterances = sum(len(v) for v in by_speaker.values())
    summary_langs = (
        {s.speaker: s.languages for s in language_summary.speakers} if (language_summary) else {}
    )

    stats: list[SpeakerStats] = []
    for speaker in set(intervals) | set(by_speaker):
        spans = intervals.get(speaker, [])
        own = by_speaker[speaker]
        speaking = total_length(spans)
        durations = [u.duration for u in own]
        words = sum(len(u.words) for u in own)
        longest = max(own, key=lambda u: (u.duration, -u.start), default=None)
        stats.append(
            SpeakerStats(
                speaker=speaker,
                speaking_time_seconds=speaking,
                speaking_time_percent_of_speech=_pct(speaking, total_speech),
                speaking_time_percent_of_meeting=_pct(speaking, meeting_duration),
                num_segments=len(own),
                segment_share_percent=_pct(len(own), total_utterances),
                num_turns=runs[speaker],
                avg_utterance_seconds=statistics.fmean(durations) if durations else 0.0,
                median_utterance_seconds=statistics.median(durations) if durations else 0.0,
                longest_utterance=(
                    TimeSpan(start=longest.start, end=longest.end, duration=longest.duration)
                    if longest
                    else None
                ),
                word_count=words,
                words_per_minute=words / (speaking / 60) if speaking > 0 else 0.0,
                first_spoke_at=spans[0][0] if spans else None,
                last_spoke_at=spans[-1][1] if spans else None,
                interruptions_made=made[speaker],
                interruptions_received=received[speaker],
                overlap_seconds=total_length(overlaps.get(speaker, [])),
                language_breakdown=summary_langs.get(speaker)
                or language_shares(utterance_language_seconds(own)),
                questions_asked=sum(is_question(u.text) for u in own),
            )
        )
    stats.sort(key=lambda s: (-s.speaking_time_seconds, s.speaker))
    return stats


def _pct(part: float, whole: float) -> float:
    return part / whole * 100 if whole > 0 else 0.0
