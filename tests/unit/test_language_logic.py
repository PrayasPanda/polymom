from dataclasses import replace

import pytest

from app.schemas.asr import TranscriptSegment, Word
from app.schemas.diarization import SpeakerTurn
from app.schemas.language import LanguageRegion
from app.services.language.base import predict, restrict_scores
from app.services.language.service import candidate_languages
from app.services.language.smoothing import (
    LidWindow,
    inherit_short_windows,
    merge_adjacent,
    resolve_uncertain,
    smooth,
    to_regions,
    windows_from_turns,
)
from app.services.language.summary import summarize, switch_points
from app.services.language.text_tagger import (
    code_mix_stats,
    script_of,
    tag_segment,
    word_language,
)

# --- scores ---------------------------------------------------------------------------


def test_restrict_scores_renormalizes_over_allowed() -> None:
    raw = {"hi": 0.3, "bn": 0.4, "or": 0.1, "en": 0.1, "ur": 0.1}

    assert restrict_scores(raw, {"en", "hi", "or"}) == pytest.approx(
        {"hi": 0.6, "or": 0.2, "en": 0.2}
    )


def test_restrict_scores_uniform_when_nothing_allowed_scored() -> None:
    assert restrict_scores({"fr": 1.0}, {"en", "hi"}) == {"en": 0.5, "hi": 0.5}
    assert restrict_scores({"fr": 1.0}, set()) == {}


def test_predict_top_k_and_uncertain() -> None:
    pred = predict({"en": 0.2, "hi": 0.45, "or": 0.35}, {"en", "hi", "or"}, 0.5)

    assert (pred.language, pred.confidence, pred.uncertain) == ("hi", 0.45, True)
    assert [s.language for s in pred.top_k] == ["hi", "or", "en"]


def test_predict_hint_filtering_changes_the_winner() -> None:
    # Hinted meeting (en + or): Hindi is not a candidate even though it scored highest.
    pred = predict({"hi": 0.7, "or": 0.2, "en": 0.1}, {"en", "or"}, 0.5)

    assert (pred.language, pred.uncertain) == ("or", False)
    assert pred.confidence == pytest.approx(0.6667, abs=1e-4)


def test_predict_requires_candidates() -> None:
    with pytest.raises(ValueError, match="no candidate"):
        predict({"en": 1.0}, set(), 0.5)


@pytest.mark.parametrize(
    ("hints", "expected"),
    [([], ["en", "hi", "or"]), (["or", "en"], ["en", "or"]), (["fr"], ["en", "hi", "or"])],
)
def test_candidate_languages(hints: list[str], expected: list[str]) -> None:
    assert candidate_languages(hints, frozenset({"en", "hi", "or"})) == expected


# --- windows -----------------------------------------------------------------------------


def turn(start: float, end: float, speaker: str) -> SpeakerTurn:
    return SpeakerTurn(
        speaker_label=speaker,
        raw_label="X",
        start=start,
        end=end,
        duration=end - start,
        is_overlap=False,
    )


def test_windows_follow_turns_and_split_long_ones() -> None:
    windows = windows_from_turns([turn(0, 4, "Person 1"), turn(4, 40, "Person 2")], max_window=15)

    assert [(w.start, w.end, w.speaker) for w in windows] == [
        (0, 4, "Person 1"),
        (4, 16, "Person 2"),
        (16, 28, "Person 2"),
        (28, 40, "Person 2"),
    ]


def test_windows_without_turns_cover_the_recording() -> None:
    windows = windows_from_turns([], max_window=10, total_duration=25)

    assert [(w.start, w.end, w.speaker) for w in windows] == [
        (0, 8.333, None),
        (8.333, 16.667, None),
        (16.667, 25, None),
    ]


# --- smoothing ---------------------------------------------------------------------------


def w(
    start: float,
    end: float,
    spk: str | None,
    lang: str | None,
    conf: float = 0.9,
    uncertain: bool = False,
) -> LidWindow:
    return LidWindow(start, end, spk, lang, conf, uncertain)


def test_short_low_confidence_window_inherits_speaker_language() -> None:
    windows = [w(0, 5, "A", "hi"), w(5, 6, "A", "en", 0.6), w(6, 12, "A", "hi")]

    assert [x.language for x in inherit_short_windows(windows, min_window=1.5)] == [
        "hi",
        "hi",
        "hi",
    ]


def test_short_but_confident_window_keeps_its_language() -> None:
    windows = [w(0, 5, "A", "hi"), w(5, 6, "A", "en", 0.95)]

    assert inherit_short_windows(windows, 1.5)[1].language == "en"


def test_short_window_uses_own_speaker_profile_only() -> None:
    # Speaker B has no other windows: nothing to inherit from, even though A is Hindi.
    windows = [w(0, 5, "A", "hi"), w(5, 6, "B", "or", 0.5)]

    assert inherit_short_windows(windows, 1.5)[1].language == "or"


def test_uncertain_resolved_by_nearest_same_speaker_window() -> None:
    windows = [
        w(0, 5, "A", "hi"),
        w(5, 10, "B", "en"),
        w(10, 12, "A", "en", 0.3, uncertain=True),  # nearest A window (0-5) is Hindi
        w(20, 30, "A", "or"),
    ]

    resolved = resolve_uncertain(windows, fallback="en")

    assert resolved[2].language == "hi"
    assert not any(x.uncertain for x in resolved)


def test_uncertain_prefers_closer_neighbor() -> None:
    windows = [w(0, 5, "A", "hi"), w(20, 22, "A", None, 0.0, True), w(23, 30, "A", "or")]

    assert resolve_uncertain(windows, "en")[1].language == "or"


def test_uncertain_falls_back_to_meeting_dominant_then_default() -> None:
    windows = [w(0, 10, "A", "or"), w(10, 11, "B", "en", 0.2, True)]
    assert resolve_uncertain(windows, fallback="en")[1].language == "or"

    all_uncertain = [w(0, 2, "A", "hi", 0.3, True)]
    assert resolve_uncertain(all_uncertain, fallback="en")[0].language == "en"


def test_merge_adjacent_same_speaker_and_language() -> None:
    merged = merge_adjacent(
        [
            w(0, 5, "A", "hi", 0.9),
            w(5.2, 10, "A", "hi", 0.5),
            w(10, 12, "B", "hi"),
            w(12, 14, "B", "or"),
        ]
    )

    assert [(x.start, x.end, x.speaker, x.language) for x in merged] == [
        (0, 10, "A", "hi"),
        (10, 12, "B", "hi"),
        (12, 14, "B", "or"),
    ]
    assert merged[0].confidence == pytest.approx((0.9 * 5 + 0.5 * 4.8) / 9.8)


def test_smooth_end_to_end() -> None:
    windows = [
        w(0, 6, "A", "hi"),
        w(6, 7, "A", "or", 0.55),  # short blip -> hi
        w(7, 12, "A", "hi"),
        w(12, 18, "B", "or", 0.3, uncertain=True),  # no confident B window -> meeting dominant
    ]

    smoothed = smooth(windows, min_window=1.5, fallback="en")

    assert [(x.start, x.end, x.speaker, x.language) for x in smoothed] == [
        (0, 12, "A", "hi"),
        (12, 18, "B", "hi"),
    ]


def test_to_regions_clips_overlapping_speech() -> None:
    regions = to_regions(
        [
            w(0, 5, "A", "hi"),
            w(4, 8, "B", "en"),
            w(6, 7, "A", "hi"),
            replace(w(8, 9, "B", None), language=None),
        ]
    )

    assert [(r.start, r.end, r.speaker, r.language) for r in regions] == [
        (0, 5, "A", "hi"),
        (5, 8, "B", "en"),
    ]


# --- summary ------------------------------------------------------------------------------


def region(start: float, end: float, lang: str, spk: str | None) -> LanguageRegion:
    return LanguageRegion(start=start, end=end, language=lang, speaker=spk, confidence=0.9)


def test_switch_points() -> None:
    regions = [
        region(0, 5, "hi", "Person 1"),
        region(5, 8, "hi", "Person 2"),
        region(8, 10, "or", "Person 2"),
        region(10, 12, "en", "Person 1"),
    ]

    points = switch_points(regions)

    assert [(p.timestamp, p.from_language, p.to_language, p.speaker) for p in points] == [
        (8, "hi", "or", "Person 2"),
        (10, "or", "en", "Person 1"),
    ]


def test_summarize_shares_and_speakers() -> None:
    summary = summarize(
        [
            region(0, 6, "hi", "Person 2"),
            region(6, 8, "en", "Person 10"),
            region(8, 10, "or", None),
        ],
        lid_model="m",
        candidates=["or", "en", "hi"],
    )

    assert [(s.language, s.duration_seconds, s.percentage) for s in summary.languages] == [
        ("hi", 6, 60.0),
        ("en", 2, 20.0),
        ("or", 2, 20.0),
    ]
    assert [s.speaker for s in summary.speakers] == ["Person 2", "Person 10", "unknown"]
    assert summary.speakers[0].dominant_language == "hi"
    assert summary.num_switches == 2
    assert summary.candidate_languages == ["en", "hi", "or"]


def test_summarize_empty() -> None:
    summary = summarize([], lid_model="m", candidates=["en"])

    assert (summary.languages, summary.speakers, summary.num_switches) == ([], [], 0)


# --- text tagging -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("word", "script", "language"),
    [
        ("मीटिंग", "Deva", "hi"),
        ("kal", "Latn", "hi-Latn"),
        ("5", None, None),
        ("baje", "Latn", "hi-Latn"),
        ("hai", "Latn", "hi-Latn"),
        ("Hai,", "Latn", "hi-Latn"),
        ("ଆଜିର", "Orya", "or"),
        ("meeting", "Latn", "en"),
        ("main", "Latn", "en"),  # English collision deliberately not in the Hindi lexicon
        ("kemiti", "Latn", "or-Latn"),
        ("Привет", "Other", None),
        ("?", None, None),
    ],
)
def test_script_and_word_language(word: str, script: str | None, language: str | None) -> None:
    assert script_of(word) == script
    assert word_language(word, script_of(word)) == language


def seg(text: str, words: list[str] | None, language: str | None = "hi") -> TranscriptSegment:
    return TranscriptSegment(
        id=0, start=0, end=4, text=text, language=language,
        words=[Word(text=t, start=0, end=1) for t in (words or [])],
        avg_confidence=0.9, backend="t",
    )  # fmt: skip


def test_tag_segment_hinglish() -> None:
    text = "मीटिंग kal 5 baje hai"
    tagged = tag_segment(seg(text, text.split()))

    assert [(x.text, x.script, x.language) for x in tagged.words] == [
        ("मीटिंग", "Deva", "hi"),
        ("kal", "Latn", "hi-Latn"),
        ("5", None, None),
        ("baje", "Latn", "hi-Latn"),
        ("hai", "Latn", "hi-Latn"),
    ]
    # Devanagari and romanized Hindi are both Hindi: one language, not code-mixed.
    assert (tagged.primary_language, tagged.languages_present, tagged.is_code_mixed) == (
        "hi",
        ["hi"],
        False,
    )


def test_tag_segment_odia_with_english_terms() -> None:
    text = "ଆଜି budget meeting ରେ ଆଲୋଚନା ହେବ"
    tagged = tag_segment(seg(text, None, language="or"))

    assert tagged.words == []  # the backend gave no words: tokens are tagged, not invented
    assert tagged.primary_language == "or"
    assert tagged.languages_present == ["en", "or"]
    assert tagged.is_code_mixed
    assert tagged.code_mix_ratio == pytest.approx(2 / 6, abs=1e-4)


def test_code_mix_tie_prefers_audio_language() -> None:
    words = [
        Word(text="a", start=0, end=1, language="en"),
        Word(text="b", start=0, end=1, language="or"),
    ]

    assert code_mix_stats(words, fallback="or").primary_language == "or"
    assert code_mix_stats(words, fallback=None).primary_language == "en"


def test_code_mix_without_language_words() -> None:
    stats = code_mix_stats([Word(text="5", start=0, end=1)], fallback="hi")

    assert (
        stats.primary_language,
        stats.languages_present,
        stats.is_code_mixed,
        stats.code_mix_ratio,
    ) == ("hi", ["hi"], False, 0.0)
