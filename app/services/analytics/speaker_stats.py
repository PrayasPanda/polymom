"""Speaker analytics: talk time, turn counts, interruptions, words per minute."""

from pathlib import Path


class SpeakerStatsCalculator:
    """Computes per-speaker statistics from aligned turns.

    TODO: implement in a later prompt.
    """

    async def compute(self, audio_path: Path) -> object:
        """Return per-speaker statistics."""
        raise NotImplementedError
