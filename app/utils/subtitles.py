"""Transcript export formats: SRT, WebVTT, plain text and Markdown."""

from collections.abc import Sequence

from app.schemas.asr import TranscriptSegment
from app.schemas.transcript import Utterance


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


# --- speaker-attributed transcript --------------------------------------------------


def speaker_label(utterance: Utterance) -> str:
    """Display name if one was set, else the diarization label."""
    return utterance.speaker_name or utterance.speaker


def _hms(seconds: float) -> str:
    return format_timestamp(seconds)[:8]


def utterances_to_text(utterances: Sequence[Utterance]) -> str:
    """``[00:01:23 - 00:01:30] Person 1: ...`` per utterance."""
    lines = [f"[{_hms(u.start)} - {_hms(u.end)}] {speaker_label(u)}: {u.text}" for u in utterances]
    return "\n".join(lines) + ("\n" if lines else "")


def utterances_to_srt(utterances: Sequence[Utterance]) -> str:
    cues = []
    for i, u in enumerate(utterances, start=1):
        end = max(u.end, u.start + 0.001)
        cues.append(
            f"{i}\n{format_timestamp(u.start)} --> {format_timestamp(end)}\n"
            f"{speaker_label(u)}: {u.text}\n"
        )
    return "\n".join(cues)


def utterances_to_vtt(utterances: Sequence[Utterance]) -> str:
    """WebVTT with ``<v Speaker>`` voice tags."""
    cues = ["WEBVTT\n"]
    for u in utterances:
        end = max(u.end, u.start + 0.001)
        start_ts = format_timestamp(u.start, decimal_marker=".")
        end_ts = format_timestamp(end, decimal_marker=".")
        cues.append(f"{start_ts} --> {end_ts}\n<v {speaker_label(u)}>{u.text}\n")
    return "\n".join(cues)


def utterances_to_markdown(utterances: Sequence[Utterance], title: str = "Transcript") -> str:
    """Readable blocks: consecutive utterances of one speaker share a heading."""
    lines = [f"# {title}", ""]
    previous: str | None = None
    for u in utterances:
        name = speaker_label(u)
        if name != previous:
            lines += [f"**{name}** · {_hms(u.start)}", ""]
            previous = name
        lines += [u.text, ""]
    return "\n".join(lines)
