"""Align ASR words with diarization segments to attribute text to speakers."""

from pathlib import Path


class Aligner:
    """Merges transcript and diarization outputs into speaker turns.

    TODO: implement in a later prompt.
    """

    async def align(self, audio_path: Path) -> object:
        """Return speaker-attributed transcript turns."""
        raise NotImplementedError
