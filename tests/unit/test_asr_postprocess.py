import unicodedata

import pytest

from app.schemas.asr import TranscriptSegment, Word
from app.services.asr.postprocess import (
    ChunkTranscript,
    FilterConfig,
    approximate_words,
    comparable,
    compression_ratio,
    filter_hallucinations,
    flag_low_confidence,
    is_filler_only,
    language_durations,
    merge_chunks,
    normalize_segment,
    normalize_text,
    renumber,
    shift,
)

CONFIG = FilterConfig(
    low_confidence_threshold=0.5, compression_ratio_threshold=2.4, no_speech_threshold=0.6
)


def seg(
    start: float,
    end: float,
    text: str,
    *,
    language: str | None = "en",
    conf: float | None = 0.9,
    no_speech: float | None = None,
    ratio: float | None = None,
    words: list[Word] | None = None,
) -> TranscriptSegment:
    return TranscriptSegment(
        id=0,
        start=start,
        end=end,
        text=text,
        language=language,
        words=words or [],
        avg_confidence=conf,
        backend="test",
        no_speech_prob=no_speech,
        compression_ratio=ratio,
    )


# --- NFC normalization / native script -----------------------------------------

# Odia sentence containing U+0B4B (canonically U+0B47 U+0B3E).
ODIA = unicodedata.normalize("NFC", "ଆମେ ନୂଆ ଯୋଜନାରେ ରାଜି")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # Devanagari nukta letters are composition exclusions: NFC is base + nukta,
        # so a model emitting precomposed U+0958 / U+095B gets normalized.
        ("\u0958", "\u0915\u093c"),  # qa
        ("\u0930\u093f\u0932\u0940\u095b", "\u0930\u093f\u0932\u0940\u091c\u093c"),  # "release"
        ("\u0928\u093c", "\u0929"),  # na + nukta composes to U+0929
        # Odia two-part vowel signs compose.
        ("\u0b15\u0b47\u0b3e", "\u0b15\u0b4b"),  # ko
        ("\u0b15\u0b47\u0b56", "\u0b15\u0b48"),  # kai
        ("\u0b15\u0b47\u0b57", "\u0b15\u0b4c"),  # kau
        # Odia rra (U+0B5C) is an exclusion too: stays dda + nukta.
        ("\u0b5c", "\u0b21\u0b3c"),
    ],
)
def test_normalize_text_indic_nfc(raw: str, expected: str) -> None:
    assert normalize_text(raw) == expected
    assert unicodedata.is_normalized("NFC", normalize_text(raw))


def test_normalize_text_round_trips_decomposed_odia_sentence() -> None:
    decomposed = unicodedata.normalize("NFD", ODIA)
    assert decomposed != ODIA

    assert normalize_text(decomposed) == ODIA


def test_normalize_never_transliterates_and_collapses_spaces() -> None:
    raw = "  \u0928\u092e\u0938\u094d\u0924\u0947 \n  \u0b13\u0b21\u0b3c\u0b3f\u0b06  hello "
    assert (
        normalize_text(raw)
        == "\u0928\u092e\u0938\u094d\u0924\u0947 \u0b13\u0b21\u0b3c\u0b3f\u0b06 hello"
    )


def test_normalize_segment_normalizes_words_and_drops_empty_words() -> None:
    raw = seg(
        0,
        1,
        unicodedata.normalize("NFD", ODIA),
        words=[
            Word(text=unicodedata.normalize("NFD", "ଯୋଜନା"), start=0, end=0.5),
            Word(text="  ", start=0.5, end=0.6),
        ],
    )

    out = normalize_segment(raw)

    assert out.text == ODIA
    assert [w.text for w in out.words] == ["ଯୋଜନା"]


def test_comparable_keeps_vowel_signs_and_strips_danda() -> None:
    assert comparable("मीटिंग, ठीक है। ଭାଷା॥ Hello!") == "मीटिंग ठीक है ଭାଷା hello"


# --- hallucination guards --------------------------------------------------------


def test_filter_drops_empty_and_punctuation_only() -> None:
    kept, dropped = filter_hallucinations([seg(0, 1, " ... "), seg(1, 2, "Real words.")], CONFIG)

    assert [s.text for s in kept] == ["Real words."]
    assert [r for _, r in dropped] == ["empty"]


@pytest.mark.parametrize(
    ("text", "no_speech", "conf", "expected_dropped"),
    [
        ("Um, hmm.", 0.9, 0.9, True),  # fillers during silence
        ("Thanks for watching!", 0.8, 0.9, True),  # classic Whisper hallucination
        ("हम्म", None, None, True),  # filler, model gives no confidence
        ("Umm", 0.1, 0.3, True),  # filler with low confidence
        ("Umm", 0.1, 0.95, False),  # a confident, real "umm" is kept
        ("Thank you.", 0.9, 0.9, False),  # not a filler: legit in meetings
    ],
)
def test_filler_only_dropped_only_in_silence(
    text: str, no_speech: float | None, conf: float | None, expected_dropped: bool
) -> None:
    _, dropped = filter_hallucinations([seg(0, 1, text, no_speech=no_speech, conf=conf)], CONFIG)

    assert bool(dropped) is expected_dropped
    if dropped:
        assert dropped[0][1] == "filler_in_silence"


def test_filter_drops_high_compression_ratio() -> None:
    looping = "we will we will we will we will we will we will we will we will"
    kept, dropped = filter_hallucinations(
        [seg(0, 5, looping), seg(5, 6, "Fine.", ratio=1.0)], CONFIG
    )

    assert compression_ratio(looping) > 2.4
    assert [r for _, r in dropped] == ["compression_ratio"]
    assert [s.text for s in kept] == ["Fine."]


def test_filter_uses_backend_compression_ratio_when_given() -> None:
    _, dropped = filter_hallucinations([seg(0, 1, "short", ratio=3.1)], CONFIG)

    assert [r for _, r in dropped] == ["compression_ratio"]


def test_filter_drops_repeated_identical_segments() -> None:
    segments = [
        seg(0, 2, "Let us move on."),
        seg(2, 4, "let us move on"),
        seg(4, 6, "Let us move on!"),
        seg(6, 8, "Next topic."),
    ]

    kept, dropped = filter_hallucinations(segments, CONFIG)

    assert [s.start for s in kept] == [0, 6]
    assert [r for _, r in dropped] == ["repeated", "repeated"]


def test_is_filler_only() -> None:
    assert is_filler_only("uh... um")
    assert is_filler_only("")
    assert not is_filler_only("um, the budget")


def test_compression_ratio_empty() -> None:
    assert compression_ratio("") == 0.0


def test_flag_low_confidence() -> None:
    flagged = flag_low_confidence(
        [seg(0, 1, "a", conf=0.4), seg(1, 2, "b", conf=0.8), seg(2, 3, "c", conf=None)], 0.5
    )

    assert [s.low_confidence for s in flagged] == [True, False, False]


# --- chunks -----------------------------------------------------------------------


def test_shift_moves_segment_and_words() -> None:
    s = seg(1.0, 2.5, "hi", words=[Word(text="hi", start=1.0, end=2.5, confidence=0.9)])

    moved = shift(s, 1800.0)

    assert (moved.start, moved.end) == (1801.0, 1802.5)
    assert (moved.words[0].start, moved.words[0].end) == (1801.0, 1802.5)
    assert shift(s, 0) is s


def test_merge_chunks_dedupes_overlap_by_time_and_fuzzy_text() -> None:
    # Chunk 1: 0-60, chunk 2: 55-115 (5 s overlap, midpoint 57.5).
    first = ChunkTranscript(
        0,
        60,
        [
            seg(0, 10, "Welcome to the meeting."),
            seg(54, 58, "The budget is approved for"),  # mid 56 -> owned by chunk 1
            seg(58.5, 60, "next"),  # mid 59.25 -> owned by chunk 2, dropped here
        ],
    )
    second = ChunkTranscript(
        55,
        115,
        [
            seg(55, 57, "is approved for"),  # mid 56 -> owned by chunk 1, dropped here
            seg(57.2, 60.5, "The budget is approved for next quarter."),  # dup, more complete
            seg(61, 70, "Any questions?"),
        ],
    )

    merged = merge_chunks([second, first])

    assert [(s.start, s.text) for s in merged] == [
        (0, "Welcome to the meeting."),
        (57.2, "The budget is approved for next quarter."),
        (61, "Any questions?"),
    ]


def test_merge_chunks_keeps_distinct_text_in_overlap() -> None:
    first = ChunkTranscript(0, 60, [seg(55, 57.4, "Alpha.")])
    second = ChunkTranscript(55, 115, [seg(57.3, 59, "Completely different.")])

    assert [s.text for s in merge_chunks([first, second])] == ["Alpha.", "Completely different."]


# --- misc -------------------------------------------------------------------------


def test_renumber_sorts_and_assigns_ids() -> None:
    out = renumber([seg(5, 6, "b"), seg(1, 2, "a")])

    assert [(s.id, s.text) for s in out] == [(0, "a"), (1, "b")]


def test_language_durations_longest_first() -> None:
    segments = [
        seg(0, 2, "a", language="en"),
        seg(2, 7, "b", language="hi"),
        seg(7, 8, "c", language=None),
    ]

    assert [(d.language, d.duration_seconds) for d in language_durations(segments)] == [
        ("hi", 5.0),
        ("en", 2.0),
        ("unknown", 1.0),
    ]


def test_approximate_words_spans_segment_proportionally() -> None:
    words = approximate_words("ab abcd", 10.0, 13.0)

    assert [w.text for w in words] == ["ab", "abcd"]
    assert words[0].start == 10.0
    assert words[0].end == pytest.approx(11.125, abs=0.001)  # 3/8 of 3 s
    assert words[1].end == 13.0
    assert all(w.confidence is None for w in words)


@pytest.mark.parametrize(("text", "start", "end"), [("", 0, 1), ("word", 1, 1)])
def test_approximate_words_degenerate(text: str, start: float, end: float) -> None:
    assert approximate_words(text, start, end) == []
