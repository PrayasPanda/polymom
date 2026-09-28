"""Pure transcript post-processing.

* NFC normalization in the native script (never transliterate),
* hallucination guards (repeats, filler-only output in silence, high compression ratio),
* low-confidence flags,
* merging chunked transcripts with overlap de-duplication,
* approximate word timings for models without word timestamps.
"""

import re
import unicodedata
import zlib
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher

from app.schemas.asr import LanguageDuration, TranscriptSegment, Word

_SPACE_RE = re.compile(r"\s+")

FILLERS: frozenset[str] = frozenset(
    {
        "uh", "um", "umm", "hmm", "hm", "mm", "mhm", "ah", "eh", "oh", "er",
        "हम्म", "उम्म", "अं", "हूं", "ହଁ", "ହମ୍",
    }
)  # fmt: skip
# Phrases Whisper is known to invent over silence (from subtitle-heavy training data).
KNOWN_HALLUCINATIONS: frozenset[str] = frozenset(
    {
        "thanks for watching",
        "thank you for watching",
        "please subscribe",
        "subscribe to my channel",
        "like and subscribe",
        "subtitles by the amaraorg community",
    }
)
MIN_CHARS_FOR_COMPRESSION_CHECK = 20
OVERLAP_TEXT_SIMILARITY = 0.8


def normalize_text(text: str) -> str:
    """NFC-normalize and collapse whitespace. Script is preserved as-is."""
    return _SPACE_RE.sub(" ", unicodedata.normalize("NFC", text)).strip()


def comparable(text: str) -> str:
    r"""Lower-cased, punctuation-free form used for duplicate/filler checks.

    Keeps letters, combining marks and digits by Unicode category, so Devanagari
    and Odia vowel signs stay attached (a ``\W`` regex would split them); dandas and
    other punctuation become spaces.
    """
    chars = (c if unicodedata.category(c)[0] in "LMN" else " " for c in normalize_text(text))
    return _SPACE_RE.sub(" ", "".join(chars).casefold()).strip()


def compression_ratio(text: str) -> float:
    """zlib compression ratio, as Whisper computes it; high values mean repetition."""
    data = text.encode("utf-8")
    return len(data) / len(zlib.compress(data)) if data else 0.0


def is_filler_only(text: str) -> bool:
    words = comparable(text).split()
    return not words or all(w in FILLERS for w in words) or " ".join(words) in KNOWN_HALLUCINATIONS


def normalize_segment(segment: TranscriptSegment) -> TranscriptSegment:
    words = [w.model_copy(update={"text": normalize_text(w.text)}) for w in segment.words]
    return segment.model_copy(
        update={"text": normalize_text(segment.text), "words": [w for w in words if w.text]}
    )


@dataclass(frozen=True, slots=True)
class FilterConfig:
    low_confidence_threshold: float
    compression_ratio_threshold: float
    no_speech_threshold: float


def filter_hallucinations(
    segments: Sequence[TranscriptSegment], config: FilterConfig
) -> tuple[list[TranscriptSegment], list[tuple[TranscriptSegment, str]]]:
    """Drop likely hallucinations. Returns ``(kept, [(dropped, reason), ...])``.

    * ``empty``: no text after normalization.
    * ``filler_in_silence``: only fillers / known hallucination phrases, and the
      model thinks it is silence (``no_speech_prob``) or is unsure (low confidence).
    * ``compression_ratio``: highly repetitive text (a Whisper looping failure).
    * ``repeated``: identical text to the previous kept segment.
    """
    kept: list[TranscriptSegment] = []
    dropped: list[tuple[TranscriptSegment, str]] = []
    for seg in sorted(segments, key=lambda s: (s.start, s.end)):
        text = comparable(seg.text)
        ratio = seg.compression_ratio
        if ratio is None and len(seg.text) >= MIN_CHARS_FOR_COMPRESSION_CHECK:
            ratio = compression_ratio(seg.text)
        silent = seg.no_speech_prob is not None and seg.no_speech_prob >= config.no_speech_threshold
        unsure = (
            seg.avg_confidence is not None and seg.avg_confidence < config.low_confidence_threshold
        )
        reason = None
        if not text:
            reason = "empty"
        elif is_filler_only(seg.text) and (silent or unsure or seg.avg_confidence is None):
            reason = "filler_in_silence"
        elif ratio is not None and ratio > config.compression_ratio_threshold:
            reason = "compression_ratio"
        elif kept and comparable(kept[-1].text) == text:
            reason = "repeated"
        if reason:
            dropped.append((seg, reason))
        else:
            kept.append(seg)
    return kept, dropped


def flag_low_confidence(
    segments: Sequence[TranscriptSegment], threshold: float
) -> list[TranscriptSegment]:
    return [
        s.model_copy(
            update={"low_confidence": s.avg_confidence is not None and s.avg_confidence < threshold}
        )
        for s in segments
    ]


def shift(segment: TranscriptSegment, offset: float) -> TranscriptSegment:
    """Move a segment (and its words) by ``offset`` seconds."""
    if not offset:
        return segment
    return segment.model_copy(
        update={
            "start": round(segment.start + offset, 3),
            "end": round(segment.end + offset, 3),
            "words": [
                w.model_copy(
                    update={"start": round(w.start + offset, 3), "end": round(w.end + offset, 3)}
                )
                for w in segment.words
            ],
        }
    )


@dataclass(frozen=True, slots=True)
class ChunkTranscript:
    """Segments of one chunk, already shifted to absolute meeting time."""

    start: float
    end: float
    segments: list[TranscriptSegment]


def _similar(a: str, b: str) -> bool:
    ca, cb = comparable(a), comparable(b)
    if not ca or not cb:
        return False
    return ca in cb or cb in ca or SequenceMatcher(None, ca, cb).ratio() >= OVERLAP_TEXT_SIMILARITY


def merge_chunks(chunks: Sequence[ChunkTranscript]) -> list[TranscriptSegment]:
    """Join chunk transcripts, removing text transcribed twice in overlap windows.

    Within an overlap, each chunk owns the half closest to its own centre: a
    segment is kept by the chunk that owns its midpoint. A segment that still
    overlaps the previously kept one in time and has near-identical text
    (containment or fuzzy ratio >= 0.8) is treated as a duplicate and dropped.
    """
    ordered = sorted(chunks, key=lambda c: c.start)
    merged: list[TranscriptSegment] = []
    for i, chunk in enumerate(ordered):
        own_start = (chunk.start + ordered[i - 1].end) / 2 if i > 0 else float("-inf")
        own_end = (ordered[i + 1].start + chunk.end) / 2 if i < len(ordered) - 1 else float("inf")
        for seg in sorted(chunk.segments, key=lambda s: s.start):
            mid = (seg.start + seg.end) / 2
            if not own_start <= mid < own_end:
                continue
            prev = merged[-1] if merged else None
            if prev and seg.start < prev.end and _similar(prev.text, seg.text):
                if len(comparable(seg.text)) > len(comparable(prev.text)):
                    merged[-1] = seg  # keep the more complete rendering
                continue
            merged.append(seg)
    return merged


def renumber(segments: Sequence[TranscriptSegment]) -> list[TranscriptSegment]:
    ordered = sorted(segments, key=lambda s: (s.start, s.end))
    return [s.model_copy(update={"id": i}) for i, s in enumerate(ordered)]


def language_durations(segments: Sequence[TranscriptSegment]) -> list[LanguageDuration]:
    totals: dict[str, float] = defaultdict(float)
    for seg in segments:
        totals[seg.language or "unknown"] += max(0.0, seg.end - seg.start)
    return [
        LanguageDuration(language=lang, duration_seconds=round(total, 3))
        for lang, total in sorted(totals.items(), key=lambda kv: -kv[1])
    ]


def approximate_words(text: str, start: float, end: float) -> list[Word]:
    """Spread a segment's time over its words in proportion to their length.

    For models that only return text: word boundaries are estimates, accurate to
    roughly the length of a word, while the segment boundaries are exact.
    """
    tokens = normalize_text(text).split()
    if not tokens or end <= start:
        return []
    weights = [len(t) + 1 for t in tokens]  # +1 approximates the inter-word gap
    total = sum(weights)
    words: list[Word] = []
    cursor = start
    for token, weight in zip(tokens, weights, strict=True):
        span = (end - start) * weight / total
        words.append(Word(text=token, start=round(cursor, 3), end=round(cursor + span, 3)))
        cursor += span
    words[-1] = words[-1].model_copy(update={"end": round(end, 3)})
    return words
