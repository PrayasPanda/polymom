"""Speaker diarization (who spoke when)."""

from pathlib import Path


class Diarizer:
    """Wraps a diarization backend such as pyannote.audio.

    TODO: implement in a later prompt.
    """

    async def diarize(self, audio_path: Path) -> object:
        """Return speaker-labelled time segments."""
        raise NotImplementedError
