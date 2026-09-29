"""Pure utterance building from speaker-attributed words.

1. Group consecutive words of the same speaker.
2. Merge fragments (< ``min_words`` words) into the nearest utterance of the
   same speaker within ``merge_gap`` seconds.
3. Split utterances longer than ``max_seconds`` after sentence punctuation
   (. ? ! । ॥ - Hindi and Odia both use the danda); if no punctuation shows up
   by ``2 * max_seconds``, cut at a word boundary so no block grows unbounded.
"""

import unicodedata
from collections import Counter
from collections.abc import Sequence

from app.schemas.diarization import SpeakerTurn
from app.schemas.transcript import (
    UNKNOWN_SPEAKER,
    AlignedWord,
    AlignmentStats,
    SpeakerTranscript,
    Utterance,
)
from app.services.language.text_tagger import code_mix_stats

SENTENCE_END = frozenset(".?!।॥")  # . ? ! । ॥
TRAILING_QUOTES = "\"')]}»”’"  # noqa: RUF001 - closing quotes after punctuation


def _is_punctuation(token: str) -> bool:
    return bool(token) and all(unicodedata.category(c)[0] == "P" for c in token)


def join_words(tokens: Sequence[str]) -> str:
    """Space-separated, except punctuation-only tokens attach to the previous word.

    Works for every script here: Latin, Devanagari and Odia all separate words
    with spaces, and "है ।" / "ହେବ ।" become "है।" / "ହେବ।".
    """
    text = ""
    for token in tokens:
        if not token:
            continue
        text += token if (not text or _is_punctuation(token)) else f" {token}"
    return text


def ends_sentence(word: AlignedWord) -> bool:
    return word.text.rstrip(TRAILING_QUOTES)[-1:] in SENTENCE_END


def group_by_speaker(words: Sequence[AlignedWord]) -> list[list[AlignedWord]]:
    groups: list[list[AlignedWord]] = []
    for w in sorted(words, key=lambda w: (w.start, w.end)):
        if groups and groups[-1][-1].speaker == w.speaker:
            groups[-1].append(w)
        else:
            groups.append([w])
    return groups


def merge_fragments(
    groups: list[list[AlignedWord]], min_words: int, merge_gap: float
) -> list[list[AlignedWord]]:
    """Fold short groups into the nearest same-speaker group within ``merge_gap``."""
    groups = [list(g) for g in groups]
    changed = True
    while changed:
        changed = False
        for i, group in enumerate(groups):
            if len(group) >= min_words:
                continue
            best: tuple[float, int] | None = None
            for j, other in enumerate(groups):
                if j == i or other[0].speaker != group[0].speaker:
                    continue
                gap = max(other[0].start - group[-1].end, group[0].start - other[-1].end, 0.0)
                if gap <= merge_gap and (best is None or (gap, j) < best):
                    best = (gap, j)
            if best is not None:
                j = best[1]
                groups[j] = sorted(groups[j] + group, key=lambda w: (w.start, w.end))
                del groups[i]
                changed = True
                break
    return sorted(groups, key=lambda g: (g[0].start, g[0].speaker))


def split_long(group: list[AlignedWord], max_seconds: float) -> list[list[AlignedWord]]:
    parts: list[list[AlignedWord]] = []
    current: list[AlignedWord] = []
    for w in group:
        current.append(w)
        span = current[-1].end - current[0].start
        if (span >= max_seconds and ends_sentence(w)) or span >= 2 * max_seconds:
            parts.append(current)
            current = []
    if current:
        parts.append(current)
    return parts


def to_utterance(words: list[AlignedWord], index: int) -> Utterance:
    languages = Counter(w.language.split("-")[0] for w in words if w.language)
    mix = code_mix_stats(words, fallback=languages.most_common(1)[0][0] if languages else None)
    confidences = [w.confidence for w in words if w.confidence is not None]
    others = sorted({s for w in words for s in w.overlapping_speakers} - {words[0].speaker})
    start, end = words[0].start, max(w.end for w in words)
    return Utterance(
        id=index,
        speaker=words[0].speaker,
        start=round(start, 3),
        end=round(end, 3),
        duration=round(end - start, 3),
        text=join_words([w.text for w in words]),
        words=words,
        primary_language=mix.primary_language,
        languages_present=mix.languages_present,
        is_code_mixed=mix.is_code_mixed,
        avg_confidence=round(sum(confidences) / len(confidences), 4) if confidences else None,
        has_overlap=bool(others),
        overlapping_speakers=others,
        alignment_precision=(
            "segment" if any(w.alignment_precision == "segment" for w in words) else "word"
        ),
    )


def build_utterances(
    words: Sequence[AlignedWord], *, max_seconds: float, min_words: int, merge_gap: float
) -> list[Utterance]:
    groups = merge_fragments(group_by_speaker(words), min_words, merge_gap)
    parts = [part for g in groups for part in split_long(g, max_seconds)]
    parts.sort(key=lambda p: (p[0].start, p[0].speaker))
    return [to_utterance(p, i) for i, p in enumerate(parts)]


def build_transcript(
    words: Sequence[AlignedWord],
    turns: Sequence[SpeakerTurn],
    *,
    total_duration: float,
    max_seconds: float,
    min_words: int,
    merge_gap: float,
) -> SpeakerTranscript:
    """Utterances plus the label-consistency sanity pass and alignment stats."""
    utterances = build_utterances(
        words, max_seconds=max_seconds, min_words=min_words, merge_gap=merge_gap
    )
    diarized = {t.speaker_label for t in turns}
    speaking = {u.speaker for u in utterances}
    warnings = [
        f"{spk} has diarization turns but no transcribed words (silent or misdiarized)."
        for spk in sorted(diarized - speaking, key=_person_key)
    ]
    total = len(words)
    unknown = sum(w.speaker == UNKNOWN_SPEAKER for w in words)
    segment_level = sum(w.alignment_precision == "segment" for w in words)
    if unknown:
        warnings.append(
            f"{unknown} word(s) matched no speaker turn and are labelled {UNKNOWN_SPEAKER}."
        )

    def pct(n: int) -> float:
        return round(100 * n / total, 2) if total else 0.0

    return SpeakerTranscript(
        utterances=utterances,
        speakers=sorted(diarized | speaking, key=_person_key),
        total_duration=round(total_duration, 3),
        warnings=warnings,
        alignment_stats=AlignmentStats(
            total_words=total,
            percent_assigned=pct(total - unknown),
            percent_unknown=pct(unknown),
            percent_segment_level=pct(segment_level),
        ),
    )


def _person_key(label: str) -> tuple[int, int, str]:
    number = label.rsplit(" ", 1)[-1]
    if label == UNKNOWN_SPEAKER:
        return (2, 0, label)
    return (0, int(number), label) if number.isdigit() else (1, 0, label)
