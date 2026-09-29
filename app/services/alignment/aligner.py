"""Pure, deterministic word-to-speaker assignment.

* Each word goes to the diarization turn with the largest temporal overlap with
  ``[word.start, word.end]``; ties go to the earlier turn, then the lower label.
  Other speakers overlapping the word are kept in ``overlapping_speakers``.
* A word overlapping no turn goes to the nearest turn within ``max_gap``
  seconds, else to ``Unknown``.
* Segments without trustworthy word timings (no words, or words without
  confidence, i.e. the Odia backend's length-based estimates) are split across
  the turns they overlap in proportion to overlap duration; those words get
  ``alignment_precision="segment"``.
"""

from collections.abc import Sequence

from app.schemas.asr import TranscriptSegment, Word
from app.schemas.diarization import SpeakerTurn
from app.schemas.transcript import UNKNOWN_SPEAKER, AlignedWord, AlignmentPrecision
from app.services.asr.postprocess import approximate_words

# Zero-length words still need a span to overlap anything.
MIN_WORD_SECONDS = 0.001


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def speaker_for_span(
    start: float, end: float, turns: Sequence[SpeakerTurn], max_gap: float
) -> tuple[str, list[str]]:
    """``(speaker, other overlapping speakers)`` for a time span."""
    end = max(end, start + MIN_WORD_SECONDS)
    per_speaker: dict[str, tuple[float, float]] = {}  # label -> (overlap, earliest turn start)
    for t in turns:
        ov = _overlap(start, end, t.start, t.end)
        if ov > 0:
            best, first = per_speaker.get(t.speaker_label, (0.0, t.start))
            per_speaker[t.speaker_label] = (best + ov, min(first, t.start))
    if per_speaker:
        ranked = sorted(per_speaker, key=lambda s: (-per_speaker[s][0], per_speaker[s][1], s))
        return ranked[0], sorted(ranked[1:])

    nearest: tuple[float, float, str] | None = None
    for t in turns:
        gap = max(t.start - end, start - t.end)
        key = (gap, t.start, t.speaker_label)
        if gap <= max_gap and (nearest is None or key < nearest):
            nearest = key
    return (nearest[2] if nearest else UNKNOWN_SPEAKER), []


def _aligned(
    word: Word, speaker: str, others: list[str], precision: AlignmentPrecision
) -> AlignedWord:
    return AlignedWord(
        **word.model_dump(),
        speaker=speaker,
        overlapping_speakers=others,
        alignment_precision=precision,
    )


def _has_word_timings(segment: TranscriptSegment) -> bool:
    return bool(segment.words) and all(w.confidence is not None for w in segment.words)


def split_segment(
    segment: TranscriptSegment, turns: Sequence[SpeakerTurn], max_gap: float
) -> list[AlignedWord]:
    """Distribute a segment's words across overlapping turns by overlap duration.

    Words keep their order; each turn's share of words is proportional to its
    overlap with the segment (largest-remainder rounding), and word times are
    spread evenly over that turn's part of the segment.
    """
    tokens = [w.text for w in segment.words] or segment.text.split()
    if not tokens:
        return []
    parts: list[tuple[float, float, str]] = []  # (start, end, speaker), chronological
    for t in sorted(turns, key=lambda t: (t.start, t.speaker_label)):
        start, end = max(segment.start, t.start), min(segment.end, t.end)
        if end > start:
            if parts and parts[-1][2] == t.speaker_label and start <= parts[-1][1]:
                parts[-1] = (parts[-1][0], max(parts[-1][1], end), t.speaker_label)
            elif not parts or start >= parts[-1][1]:
                parts.append((start, end, t.speaker_label))
            # else: overlapping speech inside the segment; the earlier turn keeps it.
    if not parts:
        speaker, _ = speaker_for_span(segment.start, segment.end, turns, max_gap)
        parts = [(segment.start, segment.end, speaker)]

    total = sum(end - start for start, end, _ in parts)
    if total > 0:
        exact = [len(tokens) * (end - start) / total for start, end, _ in parts]
    else:  # zero-length segment: everything goes to its only speaker
        exact = [float(len(tokens))] + [0.0] * (len(parts) - 1)
    counts = [int(x) for x in exact]
    for i in sorted(range(len(parts)), key=lambda i: (-(exact[i] - counts[i]), i))[
        : len(tokens) - sum(counts)
    ]:
        counts[i] += 1

    words: list[AlignedWord] = []
    cursor = 0
    for (start, end, speaker), count in zip(parts, counts, strict=True):
        chunk = tokens[cursor : cursor + count]
        timed = approximate_words(" ".join(chunk), start, end)
        if len(timed) != len(chunk):  # zero-length span: keep every word, untimed
            timed = [Word(text=t, start=start, end=end) for t in chunk]
        for i, estimate in enumerate(timed):
            original = segment.words[cursor + i] if segment.words else None
            base = (
                original.model_copy(update={"start": estimate.start, "end": estimate.end})
                if original
                else estimate
            )
            words.append(_aligned(base, speaker, [], "segment"))
        cursor += count
    return words


def align_words(
    segments: Sequence[TranscriptSegment], turns: Sequence[SpeakerTurn], max_gap: float
) -> list[AlignedWord]:
    """Every transcript word, attributed to exactly one speaker, in time order."""
    aligned: list[AlignedWord] = []
    for segment in sorted(segments, key=lambda s: (s.start, s.end, s.id)):
        if _has_word_timings(segment):
            for w in segment.words:
                speaker, others = speaker_for_span(w.start, w.end, turns, max_gap)
                aligned.append(_aligned(w, speaker, others, "word"))
        else:
            aligned.extend(split_segment(segment, turns, max_gap))
    return aligned
