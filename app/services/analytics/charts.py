"""PNG charts of the analytics, via matplotlib from the optional ``viz`` extra.

``uv sync --extra viz``. Without it :func:`render_chart` raises
:class:`ChartsUnavailableError` (HTTP 503) and everything else keeps working.
"""

import io
from collections.abc import Sequence
from typing import Any, Literal

from app.core.exceptions import ChartsUnavailableError
from app.schemas.analytics import ConversationAnalytics
from app.schemas.diarization import SpeakerTurn

ChartName = Literal["speaking-time", "timeline"]


def _figure() -> Any:
    try:
        from matplotlib.figure import Figure  # optional extra
    except ImportError as exc:
        raise ChartsUnavailableError(
            "Charts need the optional 'viz' extra: uv sync --extra viz."
        ) from exc
    return Figure(figsize=(8, 4), dpi=80, layout="constrained")


def _png(fig: Any) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    return buf.getvalue()


def speaking_time_chart(analytics: ConversationAnalytics, names: dict[str, str]) -> bytes:
    """Horizontal bars: speaking seconds per speaker, largest on top."""
    fig = _figure()
    ax = fig.add_subplot()
    speakers = list(reversed(analytics.speakers))
    ax.barh(
        [names.get(s.speaker, s.speaker) for s in speakers],
        [s.speaking_time_seconds for s in speakers],
        color="#4C72B0",
    )
    for i, s in enumerate(speakers):
        ax.annotate(
            f" {s.speaking_time_percent_of_speech:.0f}%", (s.speaking_time_seconds, i), va="center"
        )
    ax.set_xlabel("Speaking time (s)")
    ax.set_title("Speaking time per speaker")
    ax.spines[["top", "right"]].set_visible(False)
    return _png(fig)


def timeline_chart(
    analytics: ConversationAnalytics, names: dict[str, str], turns: Sequence[SpeakerTurn]
) -> bytes:
    """Gantt-style: one row per speaker, one bar per diarization turn."""
    fig = _figure()
    ax = fig.add_subplot()
    order = [s.speaker for s in analytics.speakers]
    for row, speaker in enumerate(order):
        spans = [(t.start, t.end - t.start) for t in turns if t.speaker_label == speaker]
        ax.broken_barh(spans, (row - 0.4, 0.8), color=f"C{row % 10}")
    ax.set_yticks(range(len(order)), [names.get(s, s) for s in order])
    ax.set_xlim(0, max(analytics.meeting_stats.meeting_duration_seconds, 1))
    ax.invert_yaxis()
    ax.set_xlabel("Time (s)")
    ax.set_title("Who spoke when")
    ax.spines[["top", "right"]].set_visible(False)
    return _png(fig)


def render_chart(
    name: ChartName,
    analytics: ConversationAnalytics,
    names: dict[str, str],
    turns: Sequence[SpeakerTurn],
) -> bytes:
    if name == "timeline":
        return timeline_chart(analytics, names, turns)
    return speaking_time_chart(analytics, names)
