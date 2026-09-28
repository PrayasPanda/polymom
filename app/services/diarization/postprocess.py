"""Pure post-processing of raw diarization output.

Order of operations (see :func:`build_result`):
1. merge same-speaker turns separated by short gaps,
2. drop very short turns (never the last trace of a speaker),
3. mark overlapping speech,
4. relabel speakers "Person N" by order of first appearance.

:func:`relink_chunks` stitches per-chunk results of long recordings into one
timeline with consistent speaker identities.
"""

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field, replace

from app.schemas.diarization import DiarizationResult, OverlapRegion, SpeakerTurn

Embedding = Sequence[float]


@dataclass(frozen=True, slots=True)
class RawSegment:
    """One model-emitted speech segment, in seconds."""

    start: float
    end: float
    label: str

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True, slots=True)
class RawDiarization:
    """Backend output before post-processing."""

    segments: list[RawSegment]
    embeddings: dict[str, list[float]] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ChunkDiarization:
    """Raw diarization of one chunk, with chunk-relative timestamps."""

    offset: float
    duration: float
    raw: RawDiarization


def _sorted(segments: Sequence[RawSegment]) -> list[RawSegment]:
    return sorted(segments, key=lambda s: (s.start, s.end, s.label))


def merge_gaps(segments: Sequence[RawSegment], max_gap: float) -> list[RawSegment]:
    """Merge a speaker's consecutive turns when the silence between them is < ``max_gap``.

    Merging is per speaker, so a brief interjection by someone else (an overlap)
    does not split an otherwise continuous turn.
    """
    by_speaker: dict[str, list[RawSegment]] = defaultdict(list)
    for seg in _sorted(segments):
        turns = by_speaker[seg.label]
        if turns and seg.start - turns[-1].end < max_gap:
            last = turns[-1]
            turns[-1] = replace(last, end=max(last.end, seg.end))
        else:
            turns.append(seg)
    return _sorted([t for turns in by_speaker.values() for t in turns])


def drop_short_turns(segments: Sequence[RawSegment], min_duration: float) -> list[RawSegment]:
    """Drop turns shorter than ``min_duration``.

    A speaker's only turn is always kept. If every turn of a speaker is short,
    their longest one is kept so the speaker does not vanish.
    """
    by_speaker: dict[str, list[RawSegment]] = defaultdict(list)
    for seg in segments:
        by_speaker[seg.label].append(seg)

    kept: list[RawSegment] = []
    for turns in by_speaker.values():
        long_enough = [t for t in turns if t.duration >= min_duration]
        kept.extend(long_enough or [max(turns, key=lambda t: t.duration)])
    return _sorted(kept)


def find_overlaps(segments: Sequence[RawSegment]) -> list[tuple[float, float, list[str]]]:
    """Sweep-line search for spans where two or more speakers are active."""
    events: list[tuple[float, int, str]] = []
    for seg in segments:
        if seg.end > seg.start:
            events.append((seg.start, 1, seg.label))
            events.append((seg.end, -1, seg.label))
    # Ends before starts at the same instant: touching turns do not overlap.
    events.sort(key=lambda e: (e[0], e[1]))

    active: dict[str, int] = defaultdict(int)
    regions: list[tuple[float, float, list[str]]] = []
    prev_time: float | None = None
    for time, delta, label in events:
        speaking = sorted(lbl for lbl, n in active.items() if n > 0)
        if prev_time is not None and time > prev_time and len(speaking) >= 2:
            if regions and regions[-1][1] == prev_time and regions[-1][2] == speaking:
                regions[-1] = (regions[-1][0], time, speaking)
            else:
                regions.append((prev_time, time, speaking))
        active[label] += delta
        prev_time = time
    return regions


def first_appearance_labels(segments: Sequence[RawSegment]) -> dict[str, str]:
    """Map raw labels to "Person N": whoever speaks first is Person 1."""
    mapping: dict[str, str] = {}
    for seg in _sorted(segments):
        if seg.label not in mapping:
            mapping[seg.label] = f"Person {len(mapping) + 1}"
    return mapping


def _r(value: float) -> float:
    return round(value, 3)


def build_result(
    segments: Sequence[RawSegment],
    *,
    merge_gap: float,
    min_turn: float,
    model_name: str,
    processing_time_ms: int,
) -> DiarizationResult:
    """Clean raw segments and produce the public :class:`DiarizationResult`."""
    cleaned = drop_short_turns(merge_gaps(segments, merge_gap), min_turn)
    labels = first_appearance_labels(cleaned)
    overlaps = find_overlaps(cleaned)

    def overlapping(seg: RawSegment) -> bool:
        return any(start < seg.end and seg.start < end for start, end, _ in overlaps)

    turns = [
        SpeakerTurn(
            speaker_label=labels[seg.label],
            raw_label=seg.label,
            start=_r(seg.start),
            end=_r(seg.end),
            duration=_r(seg.duration),
            is_overlap=overlapping(seg),
        )
        for seg in cleaned
    ]
    regions = [
        OverlapRegion(
            start=_r(start),
            end=_r(end),
            speakers=sorted((labels[s] for s in speakers), key=_person_number),
        )
        for start, end, speakers in overlaps
    ]
    return DiarizationResult(
        turns=turns,
        num_speakers=len(labels),
        overlap_regions=regions,
        model_name=model_name,
        processing_time_ms=processing_time_ms,
    )


def _person_number(label: str) -> int:
    return int(label.rsplit(" ", 1)[-1])


# --- Cross-chunk speaker re-linking -------------------------------------------


def cosine_similarity(a: Embedding, b: Embedding) -> float:
    """Cosine similarity; 0.0 for zero-length or non-finite vectors."""
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    if norm == 0 or not math.isfinite(dot) or not math.isfinite(norm):
        return 0.0
    return dot / norm


def _valid(embedding: Embedding | None) -> bool:
    return bool(embedding) and all(math.isfinite(x) for x in embedding or [])


def relink_chunks(chunks: Sequence[ChunkDiarization], threshold: float) -> list[RawSegment]:
    """Merge per-chunk diarizations into one timeline with global speaker labels.

    Speakers are matched to the running set of global speakers by cosine
    similarity of their embeddings (greedy, one-to-one within a chunk). A local
    speaker whose best similarity is below ``threshold``, or who has no usable
    embedding, becomes a new global speaker.

    Each chunk only "owns" half of the overlap it shares with its neighbours, so
    speech in the overlap windows is not counted twice. Timestamps are shifted
    by the chunk offset.
    """
    centroids: list[list[float] | None] = []
    weights: list[int] = []
    merged: list[RawSegment] = []
    ordered = sorted(chunks, key=lambda c: c.offset)

    for i, chunk in enumerate(ordered):
        mapping = _match_speakers(chunk.raw, centroids, weights, threshold)

        own_start = chunk.offset
        own_end = chunk.offset + chunk.duration
        if i > 0:
            prev = ordered[i - 1]
            own_start = (chunk.offset + prev.offset + prev.duration) / 2
        if i < len(ordered) - 1:
            nxt = ordered[i + 1]
            own_end = (nxt.offset + chunk.offset + chunk.duration) / 2

        for seg in chunk.raw.segments:
            start = max(seg.start + chunk.offset, own_start)
            end = min(seg.end + chunk.offset, own_end)
            if end > start:
                merged.append(RawSegment(start, end, mapping[seg.label]))
    return _sorted(merged)


def _match_speakers(
    raw: RawDiarization,
    centroids: list[list[float] | None],
    weights: list[int],
    threshold: float,
) -> dict[str, str]:
    """Assign each local label a global label, updating centroids in place."""
    local_labels = list(first_appearance_labels(raw.segments))
    candidates: list[tuple[float, str, int]] = []
    for label in local_labels:
        emb = raw.embeddings.get(label)
        if not _valid(emb):
            continue
        for g, centroid in enumerate(centroids):
            if centroid is not None:
                sim = cosine_similarity(emb or [], centroid)
                if sim >= threshold:
                    candidates.append((sim, label, g))

    mapping: dict[str, int] = {}
    taken: set[int] = set()
    for _, label, g in sorted(candidates, key=lambda c: -c[0]):
        if label not in mapping and g not in taken:
            mapping[label] = g
            taken.add(g)

    for label in local_labels:
        emb = raw.embeddings.get(label)
        if label not in mapping:
            mapping[label] = len(centroids)
            centroids.append(list(emb) if emb is not None and _valid(emb) else None)
            weights.append(1)
        elif emb is not None and _valid(emb):
            g = mapping[label]
            old = centroids[g]
            n = weights[g]
            centroids[g] = (
                [(o * n + e) / (n + 1) for o, e in zip(old, emb, strict=True)]
                if old is not None
                else list(emb)
            )
            weights[g] = n + 1
    return {label: f"SPEAKER_G{g:02d}" for label, g in mapping.items()}
