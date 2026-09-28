"""Speech recognition for English, Hindi, Odia and code-mixed speech."""

from pathlib import Path


class Transcriber:
    """Wraps an ASR backend such as Whisper, with language detection.

    TODO: implement in a later prompt.
    """

    async def transcribe(self, audio_path: Path) -> object:
        """Return word-level timestamped transcript segments."""
        raise NotImplementedError
