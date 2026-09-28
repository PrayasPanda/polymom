"""Transcript export formats: SubRip (.srt) and plain text."""

from collections.abc import Sequence

from app.schemas.asr import TranscriptSegment


def format_timestamp(seconds: float, *, decimal_marker: str = ",") -> str:
    """``HH:MM:SS,mmm`` (SRT). Negative input clamps to zero."""
    total_ms = max(0, round(seconds * 1000))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{decimal_marker}{ms:03d}"


def to_srt(segments: Sequence[TranscriptSegment]) -> str:
    """SubRip cues numbered from 1; empty input gives an empty string."""
    cues = []
    for index, seg in enumerate(sorted(segments, key=lambda s: s.start), start=1):
        end = max(seg.end, seg.start + 0.001)
        cues.append(
            f"{index}\n{format_timestamp(seg.start)} --> {format_timestamp(end)}\n{seg.text}\n"
        )
    return "\n".join(cues)


def to_text(segments: Sequence[TranscriptSegment]) -> str:
    """One line per segment: ``[HH:MM:SS.mmm] (lang) text``."""
    lines = []
    for seg in sorted(segments, key=lambda s: s.start):
        lang = f" ({seg.language})" if seg.language else ""
        lines.append(f"[{format_timestamp(seg.start, decimal_marker='.')}]{lang} {seg.text}")
    return "\n".join(lines) + ("\n" if lines else "")
