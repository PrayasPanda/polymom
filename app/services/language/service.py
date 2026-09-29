"""Language identification orchestration: windows -> LID -> smoothing -> regions."""

import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from starlette.concurrency import run_in_threadpool

from app.core.config import Settings
from app.core.logging import get_logger
from app.schemas.diarization import SpeakerTurn
from app.schemas.language import LanguageRegion, LanguageSummary
from app.services.audio.slicing import wav_duration
from app.services.language.base import LanguageIdentifier
from app.services.language.mms_lid import MMSLanguageIdentifier
from app.services.language.mock_lid import MockLanguageIdentifier
from app.services.language.smoothing import smooth, to_regions, windows_from_turns
from app.services.language.speechbrain_lid import SpeechBrainLanguageIdentifier
from app.services.language.summary import summarize
from app.services.language.whisper_lid import WhisperLanguageIdentifier

logger = get_logger(__name__)


def build_identifier(settings: Settings) -> LanguageIdentifier:
    """Identifier selected by ``LID_BACKEND``."""
    backends: dict[str, type[LanguageIdentifier]] = {
        "mms": MMSLanguageIdentifier,
        "speechbrain": SpeechBrainLanguageIdentifier,
        "whisper": WhisperLanguageIdentifier,
        "mock": MockLanguageIdentifier,
    }
    return backends[settings.lid_backend](settings)


def candidate_languages(hints: Sequence[str], supported: frozenset[str]) -> list[str]:
    """The upload's language hint is a prior: only hinted languages are considered."""
    hinted = [h for h in hints if h in supported]
    return sorted(set(hinted)) if hinted else sorted(supported)


@dataclass(frozen=True, slots=True)
class LanguageIdResult:
    regions: list[LanguageRegion]
    summary: LanguageSummary
    processing_time_ms: int


class LanguageIdService:
    def __init__(self, identifier: LanguageIdentifier, settings: Settings) -> None:
        self.identifier = identifier
        self._settings = settings

    async def identify(
        self,
        meeting_id: uuid.UUID,
        audio_path: Path,
        turns: Sequence[SpeakerTurn],
        hints: Sequence[str] = (),
    ) -> LanguageIdResult:
        settings = self._settings
        started = time.perf_counter()
        candidates = candidate_languages(hints, settings.supported_languages)
        duration = await run_in_threadpool(wav_duration, audio_path)
        windows = windows_from_turns(turns, settings.lid_max_window_seconds, duration)

        if len(candidates) == 1:
            # A single hinted language needs no model: everything is that language.
            labelled = [
                replace(w, language=candidates[0], confidence=1.0, uncertain=False) for w in windows
            ]
        else:
            labelled = []
            for w in windows:
                pred = await self.identifier.identify(audio_path, w.start, w.end, candidates)
                labelled.append(
                    replace(
                        w,
                        language=pred.language,
                        confidence=pred.confidence,
                        uncertain=pred.uncertain,
                    )
                )

        raw_uncertain = sum(w.uncertain for w in labelled)
        smoothed = smooth(
            labelled, min_window=settings.lid_min_window_seconds, fallback=candidates[0]
        )
        regions = to_regions(smoothed)
        summary = summarize(regions, lid_model=self.identifier.model_name, candidates=candidates)
        result = LanguageIdResult(
            regions=regions,
            summary=summary,
            processing_time_ms=int((time.perf_counter() - started) * 1000),
        )
        logger.info(
            "language_id_completed",
            meeting_id=str(meeting_id),
            model=self.identifier.model_name,
            candidates=candidates,
            windows=len(windows),
            uncertain_windows=raw_uncertain,
            regions=len(regions),
            switches=summary.num_switches,
            languages={s.language: s.percentage for s in summary.languages},
            processing_time_ms=result.processing_time_ms,
        )
        return result
