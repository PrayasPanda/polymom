from collections import Counter
from itertools import pairwise

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from app.schemas.asr import TranscriptSegment, Word
from app.schemas.diarization import SpeakerTurn
from app.schemas.transcript import AlignedWord, Utterance
from app.services.alignment.aligner import align_words, speaker_for_span, split_segment
from app.services.alignment.utterances import (
    build_transcript,
    build_utterances,
    group_by_speaker,
    join_words,
    merge_fragments,
    split_long,
)
from app.utils.subtitles import (
    utterances_to_markdown,
    utterances_to_srt,
    utterances_to_text,
    utterances_to_vtt,
)


def turn(start: float, end: float, speaker: str) -> SpeakerTurn:
    return SpeakerTurn(
        speaker_label=speaker,
        raw_label="X",
        start=start,
        end=end,
        duration=end - start,
        is_overlap=False,
    )


def word(
    text: str, start: float, end: float, conf: float | None = 0.9, language: str | None = "en"
) -> Word:
    return Word(text=text, start=start, end=end, confidence=conf, language=language)


def seg(
    words: list[Word], start: float, end: float, text: str | None = None, sid: int = 0
) -> TranscriptSegment:
    return TranscriptSegment(
        id=sid, start=start, end=end, text=text or " ".join(w.text for w in words),
        language="en", words=words, avg_confidence=0.9, backend="t",
    )  # fmt: skip


def aw(text: str, start: float, end: float, speaker: str, **kw: object) -> AlignedWord:
    return AlignedWord(text=text, start=start, end=end, speaker=speaker, **kw)


TURNS = [turn(0, 5, "Person 1"), turn(4, 10, "Person 2"), turn(20, 25, "Person 1")]

# --- word assignment ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("start", "end", "speaker", "others"),
    [
        (1.0, 2.0, "Person 1", []),  # inside one turn
        (4.2, 5.5, "Person 2", ["Person 1"]),  # overlap: more of it falls in Person 2
        (3.8, 4.6, "Person 1", ["Person 2"]),  # overlap: more in Person 1
        (10.5, 10.8, "Person 2", []),  # no overlap, 0.5 s after Person 2 -> nearest
        (19.4, 19.8, "Person 1", []),  # no overlap, 0.2 s before Person 1's next turn
        (14.0, 15.0, "Unknown", []),  # far from every turn
        (2.0, 2.0, "Person 1", []),  # zero-length word
    ],
)
def test_speaker_for_span(start: float, end: float, speaker: str, others: list[str]) -> None:
    assert speaker_for_span(start, end, TURNS, max_gap=1.0) == (speaker, others)


def test_ties_go_to_the_earlier_turn() -> None:
    turns = [turn(5, 10, "Person 2"), turn(0, 5, "Person 1")]

    assert speaker_for_span(4.5, 5.5, turns, 1.0) == ("Person 1", ["Person 2"])


def test_no_turns_means_unknown() -> None:
    assert speaker_for_span(0, 1, [], 1.0) == ("Unknown", [])


def test_align_words_word_level() -> None:
    segments = [seg([word("Hello", 1, 1.5), word("there", 4.3, 5.5)], 1, 5.5)]

    aligned = align_words(segments, TURNS, 1.0)

    assert [
        (w.text, w.speaker, w.overlapping_speakers, w.alignment_precision) for w in aligned
    ] == [
        ("Hello", "Person 1", [], "word"),
        ("there", "Person 2", ["Person 1"], "word"),
    ]


def test_segment_without_words_is_split_proportionally() -> None:
    # 6 tokens over 2-8 s: Person 1 has 2-4 s (2 s), Person 2 has 5-8 s after the overlap (3 s)...
    turns = [turn(0, 4, "Person 1"), turn(4, 10, "Person 2")]
    segment = seg([], 2, 8, text="ଆଜି ବଜେଟ୍ ଉପରେ ଆଲୋଚନା ହେବ ।")

    aligned = align_words([segment], turns, 1.0)

    assert [w.speaker for w in aligned] == [
        "Person 1",
        "Person 1",
        "Person 2",
        "Person 2",
        "Person 2",
        "Person 2",
    ]
    assert all(w.alignment_precision == "segment" for w in aligned)
    assert aligned[0].start == 2
    assert aligned[-1].end == 8
    assert aligned[2].start == 4


def test_words_without_confidence_are_treated_as_segment_level() -> None:
    # Odia backend: approximate word times, no confidence.
    segment = seg([word("ଆଜି", 0, 1, conf=None), word("ବଜେଟ୍", 1, 3, conf=None)], 0, 3)

    aligned = align_words([segment], [turn(0, 3, "Person 2")], 1.0)

    assert [(w.text, w.speaker, w.alignment_precision) for w in aligned] == [
        ("ଆଜି", "Person 2", "segment"),
        ("ବଜେଟ୍", "Person 2", "segment"),
    ]
    assert aligned[0].language == "en"  # word fields such as language are carried over


def test_segment_outside_turns_uses_nearest_or_unknown() -> None:
    near = align_words([seg([], 10.2, 11, text="ok then")], TURNS, 1.0)
    far = align_words([seg([], 14, 15, text="ok then")], TURNS, 1.0)

    assert {w.speaker for w in near} == {"Person 2"}
    assert {w.speaker for w in far} == {"Unknown"}


def test_split_segment_empty_text() -> None:
    assert split_segment(seg([], 0, 1, text=" "), TURNS, 1.0) == []


def test_split_segment_merges_repeated_turns_of_one_speaker() -> None:
    turns = [turn(0, 2, "Person 1"), turn(1.5, 4, "Person 1")]

    aligned = split_segment(seg([], 0, 4, text="a b c d"), turns, 1.0)

    assert {w.speaker for w in aligned} == {"Person 1"}
    assert [w.text for w in aligned] == ["a", "b", "c", "d"]


# --- utterances -------------------------------------------------------------------------


def test_group_consecutive_same_speaker() -> None:
    words = [
        aw("a", 0, 1, "Person 1"),
        aw("b", 1, 2, "Person 1"),
        aw("c", 2, 3, "Person 2"),
        aw("d", 3, 4, "Person 1"),
    ]

    assert [[w.text for w in g] for g in group_by_speaker(words)] == [["a", "b"], ["c"], ["d"]]


def test_fragment_merges_into_nearest_same_speaker_utterance() -> None:
    groups = [
        [aw("so", 0, 0.5, "Person 1"), aw("the", 0.5, 1, "Person 1"), aw("plan", 1, 2, "Person 1")],
        [aw("hmm", 2.1, 2.3, "Person 2")],  # no other Person 2 utterance: stays
        [aw("works.", 2.5, 3, "Person 1")],  # fragment, 0.5 s after Person 1's utterance
    ]

    merged = merge_fragments(groups, min_words=2, merge_gap=1.0)

    assert [[w.text for w in g] for g in merged] == [["so", "the", "plan", "works."], ["hmm"]]


def test_fragment_beyond_merge_gap_stays() -> None:
    groups = [[aw("a", 0, 1, "Person 1"), aw("b", 1, 2, "Person 1")], [aw("c", 5, 6, "Person 1")]]

    assert len(merge_fragments(groups, min_words=2, merge_gap=1.0)) == 2


@pytest.mark.parametrize("stop", [".", "?", "!", "।", "॥", '."'])
def test_split_long_at_sentence_punctuation(stop: str) -> None:
    words = [
        aw(f"w{i}" + (stop if i == 3 else ""), i * 10, i * 10 + 9, "Person 1") for i in range(6)
    ]

    parts = split_long(words, max_seconds=30)

    assert [len(p) for p in parts] == [4, 2]


def test_split_long_hard_cut_without_punctuation() -> None:
    words = [aw(f"w{i}", i * 10, i * 10 + 10, "Person 1") for i in range(10)]

    assert [len(p) for p in split_long(words, max_seconds=30)] == [6, 4]


def test_split_hindi_and_odia_sentences() -> None:
    hindi = [
        aw("बजट", 0, 10, "Person 1"),
        aw("पास", 10, 20, "Person 1"),
        aw("हुआ।", 20, 31, "Person 1"),
        aw("अब", 31, 35, "Person 1"),
    ]
    odia = [
        aw("ବଜେଟ୍", 0, 20, "Person 2"),
        aw("ପାସ୍", 20, 31, "Person 2"),
        aw("ହେଲା।", 31, 32, "Person 2"),
        aw("ଏବେ", 32, 33, "Person 2"),
    ]

    assert [len(p) for p in split_long(hindi, 30)] == [3, 1]
    assert [len(p) for p in split_long(odia, 30)] == [3, 1]


@pytest.mark.parametrize(
    ("tokens", "text"),
    [
        (["Hello", "world", "."], "Hello world."),
        (["आज", "बजट", "है", "।"], "आज बजट है।"),
        (["ଆଜି", "budget", "ଉପରେ", "?"], "ଆଜି budget ଉପରେ?"),
        (["", "ok"], "ok"),
        ([], ""),
    ],
)
def test_join_words_across_scripts(tokens: list[str], text: str) -> None:
    assert join_words(tokens) == text


def test_utterance_fields() -> None:
    words = [
        aw("मीटिंग", 0, 1, "Person 1", language="hi", confidence=0.8),
        aw(
            "budget",
            1,
            2,
            "Person 1",
            language="en",
            confidence=0.6,
            overlapping_speakers=["Person 2"],
        ),
        aw("है", 2, 3, "Person 1", language="hi", alignment_precision="segment"),
    ]

    (u,) = build_utterances(words, max_seconds=30, min_words=1, merge_gap=1)

    assert (u.speaker, u.start, u.end, u.duration, u.text) == (
        "Person 1",
        0,
        3,
        3,
        "मीटिंग budget है",
    )
    assert (u.primary_language, u.languages_present, u.is_code_mixed) == ("hi", ["en", "hi"], True)
    assert u.avg_confidence == 0.7
    assert (u.has_overlap, u.overlapping_speakers) == (True, ["Person 2"])
    assert u.alignment_precision == "segment"


def test_utterances_sorted_by_start_then_speaker() -> None:
    words = [aw("b", 0, 1, "Person 2"), aw("a", 0, 1, "Person 1"), aw("c", 2, 3, "Person 1")]

    utterances = build_utterances(words, max_seconds=30, min_words=1, merge_gap=0)

    assert [(u.id, u.speaker, u.start) for u in utterances] == [
        (0, "Person 1", 0),
        (1, "Person 2", 0),
        (2, "Person 1", 2),
    ]


def test_build_transcript_warnings_and_stats() -> None:
    turns = [turn(0, 5, "Person 1"), turn(5, 10, "Person 2"), turn(10, 12, "Person 10")]
    words = [
        aw("hi", 0, 1, "Person 1"),
        aw("there", 1, 2, "Person 1", alignment_precision="segment"),
        aw("ghost", 30, 31, "Unknown"),
        aw("yes", 10, 11, "Person 10"),
    ]

    t = build_transcript(words, turns, total_duration=31, max_seconds=30, min_words=1, merge_gap=1)

    assert t.speakers == ["Person 1", "Person 2", "Person 10", "Unknown"]
    assert t.warnings == [
        "Person 2 has diarization turns but no transcribed words (silent or misdiarized).",
        "1 word(s) matched no speaker turn and are labelled Unknown.",
    ]
    assert (t.alignment_stats.total_words, t.alignment_stats.percent_assigned) == (4, 75.0)
    assert (t.alignment_stats.percent_unknown, t.alignment_stats.percent_segment_level) == (
        25.0,
        25.0,
    )
    assert t.total_duration == 31


def test_build_transcript_empty() -> None:
    t = build_transcript([], [], total_duration=0, max_seconds=30, min_words=2, merge_gap=1)

    assert (t.utterances, t.speakers, t.warnings, t.alignment_stats.percent_assigned) == (
        [],
        [],
        [],
        0.0,
    )


def test_alignment_is_deterministic() -> None:
    segments = [
        seg([word("a", 4.4, 4.6), word("b", 4.5, 5.5)], 4.4, 5.5),
        seg([], 20, 22, text="x y z", sid=1),
    ]

    runs = {
        tuple(
            (w.text, w.speaker)
            for w in align_words(list(reversed(segments)), list(reversed(TURNS)), 1.0)
        )
        for _ in range(3)
    }

    assert len(runs) == 1


# --- renderers ----------------------------------------------------------------------------


def utt(
    uid: int, speaker: str, start: float, end: float, text: str, name: str | None = None
) -> Utterance:
    return Utterance(
        id=uid, speaker=speaker, speaker_name=name, start=start, end=end, duration=end - start,
        text=text, words=[], primary_language=None, languages_present=[], is_code_mixed=False,
        avg_confidence=None, has_overlap=False, overlapping_speakers=[], alignment_precision="word",
    )  # fmt: skip


UTTERANCES = [
    utt(0, "Person 1", 83.2, 90.4, "आज बजट पर बात करेंगे।", "Ravi"),
    utt(1, "Person 1", 90.5, 92, "Any questions?", "Ravi"),
    utt(2, "Person 2", 92.1, 95, "ହଁ, ମୋର ପ୍ରଶ୍ନ ଅଛି।"),
]


def test_render_text() -> None:
    assert (
        utterances_to_text(UTTERANCES).splitlines()[0]
        == "[00:01:23 - 00:01:30] Ravi: आज बजट पर बात करेंगे।"
    )
    assert utterances_to_text([]) == ""


def test_render_srt_and_vtt() -> None:
    srt = utterances_to_srt(UTTERANCES)
    vtt = utterances_to_vtt(UTTERANCES)

    assert srt.startswith("1\n00:01:23,200 --> 00:01:30,400\nRavi: आज बजट")
    assert "\n3\n00:01:32,100 --> 00:01:35,000\nPerson 2: ହଁ" in srt
    assert vtt.startswith("WEBVTT\n\n00:01:23.200 --> 00:01:30.400\n<v Ravi>आज बजट")


def test_render_markdown_groups_speaker_blocks() -> None:
    md = utterances_to_markdown(UTTERANCES)

    assert md.count("**Ravi**") == 1
    assert "**Ravi** · 00:01:23\n\nआज बजट पर बात करेंगे।\n\nAny questions?" in md
    assert "**Person 2** · 00:01:32" in md


# --- property-based -------------------------------------------------------------------------

turn_st = st.builds(
    lambda s, d, p: turn(s, s + d, f"Person {p}"),
    st.floats(0, 100, allow_nan=False),
    st.floats(0.1, 20, allow_nan=False),
    st.integers(1, 4),
)
word_st = st.builds(
    lambda s, d, conf: word("w", s, s + d, conf),
    st.floats(0, 120, allow_nan=False),
    st.floats(0, 2, allow_nan=False),
    st.sampled_from([0.9, None]),
)


@settings(max_examples=150, deadline=None)
@given(turns=st.lists(turn_st, max_size=8), words=st.lists(word_st, max_size=30))
def test_every_word_assigned_once_and_preserved(
    turns: list[SpeakerTurn], words: list[Word]
) -> None:
    segments = [
        seg([w], w.start, w.end, sid=i) for i, w in enumerate(sorted(words, key=lambda w: w.start))
    ]

    aligned = align_words(segments, turns, max_gap=1.0)
    transcript = build_transcript(
        aligned, turns, total_duration=130, max_seconds=10, min_words=2, merge_gap=1.0
    )

    assert len(aligned) == len(words)  # one speaker per word, nothing dropped or duplicated
    labels = {t.speaker_label for t in turns} | {"Unknown"}
    assert all(w.speaker in labels for w in aligned)
    assert sum(len(u.words) for u in transcript.utterances) == len(words)
    assert (
        Counter(id(w) for u in transcript.utterances for w in u.words).most_common(1)[0][1] <= 1
        if words
        else True
    )

    keys = [(u.start, u.speaker) for u in transcript.utterances]
    assert keys == sorted(keys)
    assert [u.id for u in transcript.utterances] == list(range(len(transcript.utterances)))
    by_speaker: dict[str, list[Utterance]] = {}
    for u in transcript.utterances:
        by_speaker.setdefault(u.speaker, []).append(u)
    for utterances in by_speaker.values():
        for a, b in pairwise(utterances):
            assert b.start >= a.end - 1e-9  # non-overlapping per speaker
