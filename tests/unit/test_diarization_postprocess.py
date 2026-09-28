import math

import pytest

from app.services.diarization.postprocess import (
    ChunkDiarization,
    RawDiarization,
    RawSegment,
    build_result,
    cosine_similarity,
    drop_short_turns,
    find_overlaps,
    first_appearance_labels,
    merge_gaps,
    relink_chunks,
)


def seg(start: float, end: float, label: str) -> RawSegment:
    return RawSegment(start, end, label)


def spans(segments: list[RawSegment]) -> list[tuple[float, float, str]]:
    return [(round(s.start, 3), round(s.end, 3), s.label) for s in segments]


# --- labels -------------------------------------------------------------------


def test_first_speaker_is_person_1_regardless_of_raw_label() -> None:
    segments = [seg(5, 6, "SPEAKER_00"), seg(0, 2, "SPEAKER_02"), seg(3, 4, "SPEAKER_01")]

    assert first_appearance_labels(segments) == {
        "SPEAKER_02": "Person 1",
        "SPEAKER_01": "Person 2",
        "SPEAKER_00": "Person 3",
    }


# --- merging ------------------------------------------------------------------


def test_merge_gaps_joins_same_speaker_below_threshold() -> None:
    segments = [seg(0, 2, "A"), seg(2.3, 4, "A"), seg(4.6, 6, "A"), seg(6, 7, "B")]

    assert spans(merge_gaps(segments, 0.5)) == [(0, 4, "A"), (4.6, 6, "A"), (6, 7, "B")]


def test_merge_gaps_is_per_speaker_across_interjections() -> None:
    segments = [seg(0, 3, "A"), seg(2.8, 3.1, "B"), seg(3.2, 6, "A")]

    assert spans(merge_gaps(segments, 0.5)) == [(0, 6, "A"), (2.8, 3.1, "B")]


def test_merge_gaps_keeps_contained_segment_end() -> None:
    assert spans(merge_gaps([seg(0, 10, "A"), seg(2, 3, "A")], 0.5)) == [(0, 10, "A")]


# --- short turns --------------------------------------------------------------


def test_drop_short_turns_keeps_only_turn_of_a_speaker() -> None:
    segments = [seg(0, 5, "A"), seg(5, 5.1, "A"), seg(6, 6.2, "B")]

    assert spans(drop_short_turns(segments, 0.3)) == [(0, 5, "A"), (6, 6.2, "B")]


def test_drop_short_turns_keeps_longest_when_all_are_short() -> None:
    segments = [seg(0, 0.1, "B"), seg(1, 1.25, "B"), seg(2, 2.05, "B")]

    assert spans(drop_short_turns(segments, 0.3)) == [(1, 1.25, "B")]


# --- overlaps -----------------------------------------------------------------


def test_find_overlaps() -> None:
    segments = [seg(0, 5, "A"), seg(4, 8, "B"), seg(6, 7, "C"), seg(8, 9, "A")]

    assert find_overlaps(segments) == [
        (4, 5, ["A", "B"]),
        (6, 7, ["B", "C"]),
    ]


def test_touching_turns_do_not_overlap() -> None:
    assert find_overlaps([seg(0, 2, "A"), seg(2, 4, "B")]) == []


def test_three_way_overlap_splits_regions() -> None:
    segments = [seg(0, 10, "A"), seg(2, 8, "B"), seg(4, 6, "C")]

    assert find_overlaps(segments) == [
        (2, 4, ["A", "B"]),
        (4, 6, ["A", "B", "C"]),
        (6, 8, ["A", "B"]),
    ]


# --- full result --------------------------------------------------------------


def test_build_result_cleans_labels_and_marks_overlaps() -> None:
    segments = [
        seg(0.5, 3.0, "SPEAKER_01"),
        seg(3.2, 5.0, "SPEAKER_01"),  # merged with previous (gap 0.2)
        seg(4.5, 9.0, "SPEAKER_00"),  # overlaps 4.5-5.0
        seg(9.5, 9.6, "SPEAKER_00"),  # short, dropped
        seg(12.0, 12.1, "SPEAKER_02"),  # short but only turn, kept
    ]

    result = build_result(
        segments, merge_gap=0.5, min_turn=0.3, model_name="m", processing_time_ms=7
    )

    assert [(t.speaker_label, t.raw_label, t.start, t.end, t.is_overlap) for t in result.turns] == [
        ("Person 1", "SPEAKER_01", 0.5, 5.0, True),
        ("Person 2", "SPEAKER_00", 4.5, 9.0, True),
        ("Person 3", "SPEAKER_02", 12.0, 12.1, False),
    ]
    assert result.turns[0].duration == 4.5
    assert result.num_speakers == 3
    assert [(r.start, r.end, r.speakers) for r in result.overlap_regions] == [
        (4.5, 5.0, ["Person 1", "Person 2"])
    ]
    assert (result.model_name, result.processing_time_ms) == ("m", 7)


def test_build_result_rounds_to_milliseconds() -> None:
    result = build_result(
        [seg(0.12345, 1.98765, "A")],
        merge_gap=0.5,
        min_turn=0.3,
        model_name="m",
        processing_time_ms=0,
    )
    turn = result.turns[0]
    assert (turn.start, turn.end, turn.duration) == (0.123, 1.988, 1.864)


def test_build_result_with_no_speech() -> None:
    result = build_result([], merge_gap=0.5, min_turn=0.3, model_name="m", processing_time_ms=0)

    assert (result.turns, result.num_speakers, result.overlap_regions) == ([], 0, [])


def test_single_speaker() -> None:
    result = build_result(
        [seg(0, 4, "SPEAKER_00"), seg(4.2, 8, "SPEAKER_00")],
        merge_gap=0.5,
        min_turn=0.3,
        model_name="m",
        processing_time_ms=0,
    )

    assert result.num_speakers == 1
    assert [(t.speaker_label, t.start, t.end) for t in result.turns] == [("Person 1", 0, 8)]


# --- cross-chunk re-linking ---------------------------------------------------

E1 = [1.0, 0.0, 0.0]
E2 = [0.0, 1.0, 0.0]
E3 = [0.0, 0.0, 1.0]


def chunk(
    offset: float, duration: float, segments: list[RawSegment], emb: dict[str, list[float]]
) -> ChunkDiarization:
    return ChunkDiarization(offset, duration, RawDiarization(segments, emb))


def test_relink_keeps_identity_when_local_labels_swap() -> None:
    # Chunk 2's model calls the second person SPEAKER_00 because they speak first there.
    chunks = [
        chunk(
            0,
            60,
            [seg(0, 30, "SPEAKER_00"), seg(30, 60, "SPEAKER_01")],
            {"SPEAKER_00": E1, "SPEAKER_01": E2},
        ),
        chunk(
            55,
            60,
            [seg(0, 30, "SPEAKER_00"), seg(30, 60, "SPEAKER_01")],
            {"SPEAKER_00": [0.1, 0.95, 0], "SPEAKER_01": [0.9, 0.1, 0]},
        ),
    ]

    merged = relink_chunks(chunks, threshold=0.6)

    assert spans(merged) == [
        (0, 30, "SPEAKER_G00"),
        (30, 57.5, "SPEAKER_G01"),  # clipped at the overlap midpoint (57.5)
        (57.5, 85, "SPEAKER_G01"),  # same person continues in chunk 2
        (85, 115, "SPEAKER_G00"),
    ]
    result = build_result(merged, merge_gap=0.5, min_turn=0.3, model_name="m", processing_time_ms=0)
    assert [(t.speaker_label, t.start, t.end) for t in result.turns] == [
        ("Person 1", 0, 30),
        ("Person 2", 30, 85),
        ("Person 1", 85, 115),
    ]


def test_relink_new_speaker_below_threshold_and_person_numbers_stay_stable() -> None:
    chunks = [
        chunk(0, 60, [seg(0, 60, "SPEAKER_00")], {"SPEAKER_00": E1}),
        chunk(
            55,
            60,
            [seg(0, 30, "SPEAKER_00"), seg(30, 60, "SPEAKER_01")],
            {"SPEAKER_00": E3, "SPEAKER_01": E1},
        ),
        chunk(110, 60, [seg(0, 60, "SPEAKER_00")], {"SPEAKER_00": [0.05, 0, 1]}),
    ]

    result = build_result(
        relink_chunks(chunks, threshold=0.6),
        merge_gap=0.5,
        min_turn=0.3,
        model_name="m",
        processing_time_ms=0,
    )

    assert result.num_speakers == 2
    assert [(t.speaker_label, t.start, t.end) for t in result.turns] == [
        ("Person 1", 0, 57.5),
        ("Person 2", 57.5, 85),
        ("Person 1", 85, 112.5),
        ("Person 2", 112.5, 170),  # third chunk re-linked to the chunk-2 newcomer
    ]


def test_relink_matches_one_to_one_within_a_chunk() -> None:
    # Both local speakers resemble global speaker 0; only the closer one may take it.
    chunks = [
        chunk(0, 10, [seg(0, 10, "A")], {"A": E1}),
        chunk(10, 10, [seg(0, 5, "X"), seg(5, 10, "Y")], {"X": [0.8, 0.6, 0], "Y": [0.99, 0.1, 0]}),
    ]

    merged = relink_chunks(chunks, threshold=0.5)

    assert spans(merged) == [
        (0, 10, "SPEAKER_G00"),
        (10, 15, "SPEAKER_G01"),
        (15, 20, "SPEAKER_G00"),
    ]


def test_relink_without_embeddings_creates_new_speakers() -> None:
    chunks = [
        chunk(0, 10, [seg(0, 10, "A")], {}),
        chunk(10, 10, [seg(0, 10, "A")], {"A": [math.nan, 0, 0]}),
    ]

    assert [s.label for s in relink_chunks(chunks, threshold=0.5)] == ["SPEAKER_G00", "SPEAKER_G01"]


def test_relink_drops_segments_outside_owned_window() -> None:
    # A segment entirely inside chunk 1's half of the overlap is dropped from chunk 2.
    chunks = [
        chunk(0, 60, [seg(0, 60, "A")], {"A": E1}),
        chunk(55, 60, [seg(0, 2, "B"), seg(10, 60, "A")], {"A": E1, "B": E2}),
    ]

    assert spans(relink_chunks(chunks, threshold=0.6)) == [
        (0, 57.5, "SPEAKER_G00"),
        (65, 115, "SPEAKER_G00"),
    ]


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        (E1, E1, 1.0),
        (E1, E2, 0.0),
        ([1.0, 1.0], [-1.0, -1.0], -1.0),
        ([0.0, 0.0], [1.0, 0.0], 0.0),
        ([1.0], [1.0, 0.0], 0.0),
        ([], [], 0.0),
        ([math.inf, 0.0], [1.0, 0.0], 0.0),
    ],
)
def test_cosine_similarity(a: list[float], b: list[float], expected: float) -> None:
    assert cosine_similarity(a, b) == pytest.approx(expected)
