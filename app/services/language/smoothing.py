"""Pure LID window construction and smoothing.

Windows come from diarization turns: people usually switch language at turn
boundaries, and turns are short enough for reliable LID. Raw per-window
predictions are then cleaned up:

1. :func:`inherit_short_windows` - a short, low-confidence window that disagrees
   with its speaker's dominant language takes the dominant language.
2. :func:`resolve_uncertain` - uncertain windows take the language of the nearest
   confident window of the same speaker, else the meeting's dominant language.
3. :func:`merge_adjacent` - contiguous windows of the same speaker and language
   are merged.

:func:`to_regions` finally makes the timeline non-overlapping for ASR routing.
"""

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, replace

from app.schemas.diarization import SpeakerTurn
from app.schemas.language import LanguageRegion

# A short window this confident keeps its own language even if it disagrees.
SHORT_WINDOW_OVERRIDE_CONFIDENCE = 0.75
# Windows closer than this (seconds) count as contiguous when merging.
MERGE_GAP_SECONDS = 0.5


@dataclass(frozen=True, slots=True)
class LidWindow:
    start: float
    end: float
    speaker: str | None
    language: str | None = None
    confidence: float = 0.0
    uncertain: bool = True

    @property
    def duration(self) -> float:
        return self.end - self.start


def windows_from_turns(
    turns: Sequence[SpeakerTurn], max_window: float, total_duration: float | None = None
) -> list[LidWindow]:
    """One LID window per diarization turn, splitting turns longer than ``max_window``.

    Without turns (no diarization, or no speech found) the whole recording is cut
    into ``max_window`` pieces with no speaker.
    """
    spans: list[tuple[float, float, str | None]] = [
        (t.start, t.end, t.speaker_label) for t in turns if t.end > t.start
    ]
    if not spans and total_duration:
        spans = [(0.0, total_duration, None)]
    windows: list[LidWindow] = []
    for start, end, speaker in sorted(spans, key=lambda s: (s[0], s[1])):
        parts = max(1, math.ceil((end - start) / max_window - 1e-9))
        step = (end - start) / parts
        windows.extend(
            LidWindow(round(start + i * step, 3), round(start + (i + 1) * step, 3), speaker)
            for i in range(parts)
        )
    return windows


def _dominant(windows: Sequence[LidWindow]) -> str | None:
    """Language with the most confident speech time."""
    totals: dict[str, float] = defaultdict(float)
    for w in windows:
        if w.language and not w.uncertain:
            totals[w.language] += w.duration
    return max(sorted(totals), key=lambda lang: totals[lang]) if totals else None


def speaker_profiles(windows: Sequence[LidWindow]) -> dict[str | None, str]:
    """Dominant language per speaker, from confident windows."""
    by_speaker: dict[str | None, list[LidWindow]] = defaultdict(list)
    for w in windows:
        by_speaker[w.speaker].append(w)
    profiles = {spk: _dominant(ws) for spk, ws in by_speaker.items()}
    return {spk: lang for spk, lang in profiles.items() if lang}


def inherit_short_windows(windows: Sequence[LidWindow], min_window: float) -> list[LidWindow]:
    """Short (< ``min_window``) low-confidence windows adopt their speaker's dominant language.

    The profile is computed from the speaker's *other* confident windows, so a
    short window never votes for itself.
    """
    result: list[LidWindow] = []
    for i, w in enumerate(windows):
        if (
            w.language
            and w.duration < min_window
            and w.confidence < SHORT_WINDOW_OVERRIDE_CONFIDENCE
        ):
            others = [o for j, o in enumerate(windows) if j != i and o.speaker == w.speaker]
            dominant = _dominant(others)
            if dominant and dominant != w.language:
                w = replace(w, language=dominant, uncertain=False)
        result.append(w)
    return result


def resolve_uncertain(windows: Sequence[LidWindow], fallback: str) -> list[LidWindow]:
    """Give every uncertain window a language.

    Nearest confident window of the same speaker (by time gap; ties prefer the
    earlier one), then the meeting's dominant language, then ``fallback``.
    """
    meeting = _dominant(windows) or fallback
    result: list[LidWindow] = []
    for i, w in enumerate(windows):
        if not w.uncertain and w.language:
            result.append(w)
            continue
        best: tuple[float, int, str] | None = None
        for j, o in enumerate(windows):
            if j == i or o.speaker != w.speaker or o.uncertain or not o.language:
                continue
            gap = max(0.0, o.start - w.end, w.start - o.end)
            key = (gap, 0 if o.start < w.start else 1, o.language)
            if best is None or key[:2] < best[:2]:
                best = key
        language = best[2] if best else meeting
        result.append(replace(w, language=language, uncertain=False))
    return result


def merge_adjacent(
    windows: Sequence[LidWindow], max_gap: float = MERGE_GAP_SECONDS
) -> list[LidWindow]:
    """Merge time-contiguous windows with the same speaker and language.

    Confidence becomes the duration-weighted mean.
    """
    merged: list[LidWindow] = []
    for w in sorted(windows, key=lambda x: (x.start, x.end)):
        prev = merged[-1] if merged else None
        if (
            prev
            and prev.speaker == w.speaker
            and prev.language == w.language
            and w.start - prev.end <= max_gap
        ):
            total = prev.duration + w.duration
            conf = (
                (prev.confidence * prev.duration + w.confidence * w.duration) / total
                if total
                else prev.confidence
            )
            merged[-1] = replace(prev, end=max(prev.end, w.end), confidence=conf)
        else:
            merged.append(w)
    return merged


def smooth(windows: Sequence[LidWindow], *, min_window: float, fallback: str) -> list[LidWindow]:
    """Full smoothing: inherit short windows, resolve uncertain ones, merge."""
    ordered = sorted(windows, key=lambda x: (x.start, x.end))
    return merge_adjacent(resolve_uncertain(inherit_short_windows(ordered, min_window), fallback))


def to_regions(windows: Sequence[LidWindow]) -> list[LanguageRegion]:
    """Non-overlapping, chronologically sorted regions for ASR routing.

    Overlapping speech (two speakers at once) would otherwise be transcribed
    twice; a later window is clipped to start where the earlier one ends.
    """
    regions: list[LanguageRegion] = []
    cursor = float("-inf")
    for w in sorted(windows, key=lambda x: (x.start, x.end)):
        start = max(w.start, cursor)
        if w.end - start <= 1e-6 or not w.language:
            continue
        regions.append(
            LanguageRegion(
                start=round(start, 3),
                end=round(w.end, 3),
                language=w.language,
                speaker=w.speaker,
                confidence=round(w.confidence, 4),
            )
        )
        cursor = w.end
    return regions
