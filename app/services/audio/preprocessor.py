"""Audio preprocessing: decode, resample to 16 kHz mono, normalise loudness."""

from pathlib import Path


class AudioPreprocessor:
    """Converts arbitrary audio/video input into a canonical WAV via ffmpeg.

    TODO: implement in a later prompt.
    """

    async def preprocess(self, audio_path: Path) -> object:
        """Return the path to a normalised 16 kHz mono WAV file."""
        raise NotImplementedError
