"""Energy-based speech region detection (pure; energies are computed by the caller).

Used by backends that transcribe text only (no timestamps): each detected
region becomes one segment whose boundaries are accurate to one frame.
"""

from collections.abc import Sequence

FRAME_SECONDS = 0.03
SPEECH_THRESHOLD_DB = -45.0
MIN_SILENCE_SECONDS = 0.5
MIN_REGION_SECONDS = 0.3
MAX_REGION_SECONDS = 20.0


def speech_regions(
    energies_db: Sequence[float],
    frame_seconds: float = FRAME_SECONDS,
    threshold_db: float = SPEECH_THRESHOLD_DB,
    min_silence: float = MIN_SILENCE_SECONDS,
    min_region: float = MIN_REGION_SECONDS,
    max_region: float = MAX_REGION_SECONDS,
) -> list[tuple[float, float]]:
    """Return ``(start, end)`` speech spans in seconds.

    Frames at or above ``threshold_db`` are speech; pauses shorter than
    ``min_silence`` are bridged; regions shorter than ``min_region`` are dropped;
    regions longer than ``max_region`` are split into equal parts so each fits
    the model's input window.
    """
    raw: list[tuple[int, int]] = []
    start: int | None = None
    silent_run = 0
    bridge = max(1, round(min_silence / frame_seconds))
    for i, energy in enumerate(energies_db):
        if energy >= threshold_db:
            if start is None:
                start = i
            silent_run = 0
        elif start is not None:
            silent_run += 1
            if silent_run >= bridge:
                raw.append((start, i - silent_run + 1))
                start, silent_run = None, 0
    if start is not None:
        raw.append((start, len(energies_db) - silent_run))

    regions: list[tuple[float, float]] = []
    for first, last in raw:
        begin, end = first * frame_seconds, last * frame_seconds
        if end - begin < min_region:
            continue
        parts = max(1, -int(-(end - begin) // max_region))
        step = (end - begin) / parts
        regions.extend(
            (round(begin + k * step, 3), round(begin + (k + 1) * step, 3)) for k in range(parts)
        )
    return regions
