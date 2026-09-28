"""Pure meeting-level language statistics: shares, speaker breakdown, switch points."""

from collections import defaultdict
from collections.abc import Sequence
from itertools import pairwise

from app.schemas.language import (
    LanguageRegion,
    LanguageShare,
    LanguageSummary,
    SpeakerLanguages,
    SwitchPoint,
)


def _shares(totals: dict[str, float]) -> list[LanguageShare]:
    grand = sum(totals.values())
    return [
        LanguageShare(
            language=lang,
            duration_seconds=round(seconds, 3),
            percentage=round(100 * seconds / grand, 2) if grand else 0.0,
        )
        for lang, seconds in sorted(totals.items(), key=lambda kv: (-kv[1], kv[0]))
    ]


def switch_points(regions: Sequence[LanguageRegion]) -> list[SwitchPoint]:
    """Every chronological change of language, attributed to the incoming speaker."""
    ordered = sorted(regions, key=lambda r: (r.start, r.end))
    return [
        SwitchPoint(
            timestamp=cur.start,
            from_language=prev.language,
            to_language=cur.language,
            speaker=cur.speaker,
        )
        for prev, cur in pairwise(ordered)
        if cur.language != prev.language
    ]


def summarize(
    regions: Sequence[LanguageRegion],
    *,
    lid_model: str,
    candidates: Sequence[str],
    code_mixed_segments: int = 0,
) -> LanguageSummary:
    totals: dict[str, float] = defaultdict(float)
    per_speaker: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for r in regions:
        seconds = max(0.0, r.end - r.start)
        totals[r.language] += seconds
        per_speaker[r.speaker or "unknown"][r.language] += seconds

    speakers = []
    for speaker in sorted(per_speaker, key=lambda s: (s == "unknown", _person_key(s))):
        shares = _shares(per_speaker[speaker])
        speakers.append(
            SpeakerLanguages(
                speaker=speaker, dominant_language=shares[0].language, languages=shares
            )
        )
    points = switch_points(regions)
    return LanguageSummary(
        languages=_shares(totals),
        speakers=speakers,
        num_switches=len(points),
        switch_points=points,
        code_mixed_segments=code_mixed_segments,
        regions=sorted(regions, key=lambda r: (r.start, r.end)),
        lid_model=lid_model,
        candidate_languages=sorted(candidates),
    )


def _person_key(label: str) -> tuple[int, str]:
    number = label.rsplit(" ", 1)[-1]
    return (int(number), label) if number.isdigit() else (10**9, label)
