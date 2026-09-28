"""Minutes-of-Meeting pipeline orchestrator.

Stages: preprocess -> diarize -> transcribe -> align -> analytics -> summarize.
"""

from pathlib import Path

from app.services.alignment.aligner import Aligner
from app.services.analytics.speaker_stats import SpeakerStatsCalculator
from app.services.asr.transcriber import Transcriber
from app.services.audio.preprocessor import AudioPreprocessor
from app.services.diarization.diarizer import Diarizer
from app.services.summarization.summarizer import Summarizer


class MoMPipeline:
    """Coordinates the service stages for a single meeting.

    TODO: wire stages together, persist intermediate artefacts, report progress
    and handle partial failures.
    """

    def __init__(
        self,
        preprocessor: AudioPreprocessor,
        diarizer: Diarizer,
        transcriber: Transcriber,
        aligner: Aligner,
        stats: SpeakerStatsCalculator,
        summarizer: Summarizer,
    ) -> None:
        self.preprocessor = preprocessor
        self.diarizer = diarizer
        self.transcriber = transcriber
        self.aligner = aligner
        self.stats = stats
        self.summarizer = summarizer

    async def run(self, meeting_id: str, audio_path: Path) -> None:
        """Process one meeting end to end."""
        raise NotImplementedError
