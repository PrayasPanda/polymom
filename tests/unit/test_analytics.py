"""Speaker and meeting analytics. Expected values are worked out by hand in the comments."""

import sys

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from app.core.exceptions import ChartsUnavailableError
from app.pipelines.mom_pipeline import AnalyticsStage, PipelineContext
from app.schemas.analytics import ConversationAnalytics
from app.schemas.diarization import SpeakerTurn
from app.schemas.language import LanguageShare, LanguageSummary, SpeakerLanguages
from app.schemas.transcript import (
    UNKNOWN_SPEAKER,
    AlignedWord,
    AlignmentStats,
    SpeakerTranscript,
    Utterance,
)
from app.services.analytics import charts
from app.services.analytics.export import speakers_to_csv
from app.services.analytics.meeting_stats import build_analytics, gini, participation_balance
from app.services.analytics.speaker_stats import interruptions, is_question, merge_intervals


def turn(speaker: str, start: float, end: float) -> SpeakerTurn:
    return SpeakerTurn(
        speaker_label=speaker,
        raw_label=speaker,
        start=start,
        end=end,
        duration=end - start,
        is_overlap=False,
    )


def utt(i: int, speaker: str, start: float, end: float, text: str, lang: str = "en") -> Utterance:
    words = [AlignedWord(text=w, start=start, end=end, speaker=speaker) for w in text.split()]
    return Utterance(
        id=i,
        speaker=speaker,
        start=start,
        end=end,
        duration=end - start,
        text=text,
        words=words,
        primary_language=lang,
        languages_present=[lang],
        is_code_mixed=False,
        avg_confidence=None,
        has_overlap=False,
        overlapping_speakers=[],
        alignment_precision="word",
    )


def transcript(utterances: list[Utterance], duration: float) -> SpeakerTranscript:
    return SpeakerTranscript(
        utterances=utterances,
        speakers=sorted({u.speaker for u in utterances}),
        total_duration=duration,
        warnings=[],
        alignment_stats=AlignmentStats(
            total_words=0, percent_assigned=100, percent_unknown=0, percent_segment_level=0
        ),
    )


def analyze(
    turns: list[SpeakerTurn],
    utterances: list[Utterance],
    duration: float,
    bucket: float = 10,
    summary: LanguageSummary | None = None,
) -> ConversationAnalytics:
    return build_analytics(
        turns,
        transcript(utterances, duration),
        language_summary=summary,
        interruption_min_overlap=0.5,
        bucket_seconds=bucket,
        gini_balanced_max=0.2,
        gini_dominated_min=0.4,
    )


# P1 0-10, P2 8-15 (starts inside P1, overlaps 2 s), P1 15-20, P3 30-31; meeting 40 s.
TURNS = [turn("Person 1", 0, 10), turn("Person 2", 8, 15), turn("Person 1", 15, 20)]
TURNS.append(turn("Person 3", 30, 31))
UTTERANCES = [
    utt(0, "Person 1", 0, 4, "hello everyone welcome"),
    utt(1, "Person 1", 4, 10, "what is the budget?"),
    utt(2, "Person 2", 8, 15, "क्या हम शुरू करें", "hi"),
    utt(3, "Person 1", 15, 20, "ok"),
    utt(4, UNKNOWN_SPEAKER, 25, 26, "mumble"),
]


@pytest.fixture
def report() -> ConversationAnalytics:
    return analyze(TURNS, UTTERANCES, 40)


def test_speaking_time_counts_overlap_for_both_speakers(report: ConversationAnalytics) -> None:
    p1, p2, p3 = report.speakers
    assert [s.speaker for s in report.speakers] == ["Person 1", "Person 2", "Person 3"]
    assert (p1.speaking_time_seconds, p2.speaking_time_seconds, p3.speaking_time_seconds) == (
        15,
        7,
        1,
    )
    # union of speech = [0,20] + [30,31] = 21 s; overlap [8,10] = 2 s; 15+7+1 = 21+2
    m = report.meeting_stats
    assert m.total_speech_seconds == 21
    assert m.overlap_seconds == 2
    assert m.overlap_double_counted_seconds == 2
    assert m.overlap_percent == 9.52
    assert (p1.overlap_seconds, p2.overlap_seconds, p3.overlap_seconds) == (2, 2, 0)
    assert p1.speaking_time_percent_of_speech == 71.43  # 15/21
    assert p1.speaking_time_percent_of_meeting == 37.5  # 15/40
    assert m.total_silence_seconds == 19
    assert m.silence_percent == 47.5


def test_turns_interruptions_and_monologue(report: ConversationAnalytics) -> None:
    p1, p2, p3 = report.speakers
    # floor: P1 -> P2 -> P1 -> P3
    assert (p1.num_turns, p2.num_turns, p3.num_turns) == (2, 1, 1)
    assert report.meeting_stats.total_turn_switches == 3
    assert (p2.interruptions_made, p1.interruptions_received) == (1, 1)
    assert (p1.interruptions_made, p2.interruptions_received) == (0, 0)
    mono = report.meeting_stats.longest_monologue
    assert mono is not None
    assert (mono.speaker, mono.start, mono.end, mono.duration) == ("Person 1", 0, 10, 10)
    assert (p1.first_spoke_at, p1.last_spoke_at) == (0, 20)


def test_words_segments_and_questions(report: ConversationAnalytics) -> None:
    p1, p2, p3 = report.speakers
    assert (p1.num_segments, p2.num_segments, p3.num_segments) == (3, 1, 0)
    assert (p1.segment_share_percent, p2.segment_share_percent) == (75, 25)  # Unknown excluded
    assert p1.word_count == 8
    assert p1.words_per_minute == 32  # 8 words / 0.25 min
    assert p2.words_per_minute == 34.29  # 4 / (7/60)
    assert (p1.avg_utterance_seconds, p1.median_utterance_seconds) == (5, 5)
    assert p1.longest_utterance is not None
    assert (p1.longest_utterance.start, p1.longest_utterance.end) == (4, 10)
    assert (p1.questions_asked, p2.questions_asked) == (1, 1)
    m = report.meeting_stats
    assert (m.total_utterances, m.total_words) == (4, 12)


def test_speaker_with_turns_but_no_words(report: ConversationAnalytics) -> None:
    p3 = report.speakers[2]
    assert p3.word_count == 0
    assert p3.words_per_minute == 0
    assert p3.longest_utterance is None
    assert p3.avg_utterance_seconds == 0
    assert p3.language_breakdown == []


def test_gini_balance_and_dominance(report: ConversationAnalytics) -> None:
    # sum |xi - xj| over ordered pairs of [15, 7, 1] = 2 * (8 + 14 + 6) = 56; 56 / (2*3*23)
    assert report.meeting_stats.gini_coefficient == 0.41
    assert report.meeting_stats.participation_balance == "dominated"
    assert report.meeting_stats.dominant_speaker == "Person 1"
    assert report.meeting_stats.least_active_speaker == "Person 3"


def test_languages_from_utterances(report: ConversationAnalytics) -> None:
    m = report.meeting_stats
    assert [(x.language, x.duration_seconds, x.percentage) for x in m.language_distribution] == [
        ("en", 15, 68.18),
        ("hi", 7, 31.82),
    ]
    assert m.num_language_switches == 2  # en en hi en
    assert report.speakers[1].language_breakdown[0].language == "hi"


def test_languages_from_language_summary() -> None:
    share = LanguageShare(language="or", duration_seconds=15, percentage=100)
    summary = LanguageSummary(
        languages=[share],
        speakers=[SpeakerLanguages(speaker="Person 1", dominant_language="or", languages=[share])],
        num_switches=5,
        switch_points=[],
        regions=[],
        lid_model="mock",
        candidate_languages=["or"],
    )
    report = analyze(TURNS, UTTERANCES, 40, summary=summary)
    assert report.meeting_stats.language_distribution == [share]
    assert report.meeting_stats.num_language_switches == 5
    assert report.speakers[0].language_breakdown == [share]
    assert report.speakers[1].language_breakdown[0].language == "hi"  # falls back to utterances


def test_timeline_buckets(report: ConversationAnalytics) -> None:
    assert [(b.start, b.end, b.speakers) for b in report.timeline] == [
        (0, 10, {"Person 1": 10, "Person 2": 2, "Person 3": 0}),
        (10, 20, {"Person 1": 5, "Person 2": 5, "Person 3": 0}),
        (20, 30, {"Person 1": 0, "Person 2": 0, "Person 3": 0}),
        (30, 40, {"Person 1": 0, "Person 2": 0, "Person 3": 1}),
    ]
    assert len(analyze(TURNS, UTTERANCES, 45).timeline) == 5  # last bucket is partial
    assert analyze(TURNS, UTTERANCES, 45).timeline[-1].end == 45


@pytest.mark.parametrize(
    ("values", "expected", "label"),
    [
        ([10, 10], 0.0, "balanced"),
        ([3, 1], 0.25, "moderately dominated"),  # 2*2 / (2*2*4)
        ([1, 0], 0.5, "dominated"),
        ([5], 0.0, "not applicable"),
        ([], 0.0, "not applicable"),
    ],
)
def test_gini_labels(values: list[float], expected: float, label: str) -> None:
    g = gini(values)
    assert g == pytest.approx(expected)
    assert participation_balance(g, len(values), 0.2, 0.4) == label


def test_interruption_threshold() -> None:
    turns = [turn("A", 0, 10), turn("B", 9.7, 12)]  # 0.3 s overlap
    assert interruptions(turns, 0.5) == []
    assert interruptions(turns, 0.2) == [("B", "A")]
    # starting at the same time is not an interruption; nor is same-speaker overlap
    assert interruptions([turn("A", 0, 5), turn("B", 0, 5), turn("A", 1, 3)], 0.5) == []


def test_same_speaker_overlapping_turns_counted_once() -> None:
    report = analyze([turn("A", 0, 5), turn("A", 3, 8)], [], 8)
    assert report.speakers[0].speaking_time_seconds == 8
    assert report.speakers[0].num_turns == 1
    assert merge_intervals([(3, 8), (0, 5), (9, 9)]) == [(0, 8)]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Are we done?", True),
        ("We are done.", False),
        ("Is it ready ?  ", True),
        ("आप कब आएंगे", True),
        ("क्या।", True),
        ("यह ठीक है।", False),
        ("ଆପଣ କେବେ ଆସିବେ", True),
        ("ଏହା କ’ଣ", True),  # noqa: RUF001 - Odia apostrophe
        ("ଏହା ଭଲ।", False),
    ],
)
def test_question_detection(text: str, expected: bool) -> None:
    assert is_question(text) is expected


def test_no_speech() -> None:
    report = analyze([], [], 10, bucket=60)
    m = report.meeting_stats
    assert report.speakers == []
    assert (m.total_speech_seconds, m.total_silence_seconds, m.silence_percent) == (0, 10, 100)
    assert m.participation_balance == "not applicable"
    assert (m.dominant_speaker, m.least_active_speaker, m.longest_monologue) == (None, None, None)
    assert [(b.start, b.end, b.speakers) for b in report.timeline] == [(0, 10, {})]
    assert analyze([], [], 0).timeline == []


def test_single_speaker_long_monologue() -> None:
    report = analyze(
        [turn("Person 1", 0, 3600)], [utt(0, "Person 1", 0, 3600, "a b c d")], 3600, 60
    )
    (p1,) = report.speakers
    m = report.meeting_stats
    assert p1.speaking_time_percent_of_speech == 100
    assert p1.speaking_time_percent_of_meeting == 100
    assert p1.words_per_minute == round(4 / 60, 2)
    assert m.participation_balance == "not applicable"
    assert m.longest_monologue is not None
    assert m.longest_monologue.duration == 3600
    assert m.total_turn_switches == 0
    assert len(report.timeline) == 60
    assert all(b.speakers == {"Person 1": 60} for b in report.timeline)


def test_csv_uses_display_names(report: ConversationAnalytics) -> None:
    report.speakers[0].speaker_name = "Ravi"
    rows = speakers_to_csv(report).splitlines()
    assert rows[0].startswith("speaker,display_name,speaking_time_seconds,")
    assert rows[1].startswith("Person 1,Ravi,15.0,")
    assert rows[1].endswith(",en:100.0")
    assert rows[3].startswith("Person 3,Person 3,1.0,")
    assert len(rows) == 4


def test_charts_render_png(report: ConversationAnalytics) -> None:
    pytest.importorskip("matplotlib")
    for name in ("speaking-time", "timeline"):
        png = charts.render_chart(name, report, {"Person 1": "Ravi"}, TURNS)  # type: ignore[arg-type]
        assert png.startswith(b"\x89PNG")


def test_charts_without_matplotlib(
    report: ConversationAnalytics, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "matplotlib.figure", None)
    with pytest.raises(ChartsUnavailableError):
        charts.render_chart("speaking-time", report, {}, [])


async def test_stage_requires_alignment(tmp_path: object) -> None:
    from pathlib import Path
    from uuid import uuid4

    from app.core.config import Settings

    stage = AnalyticsStage(Settings(_env_file=None))
    with pytest.raises(RuntimeError, match="AlignmentStage"):
        await stage.run(PipelineContext(meeting_id=uuid4(), input_path=Path("x.wav")))


# --- property tests ---

turn_strategy = st.builds(
    lambda spk, start, length: turn(f"Person {spk}", start, start + length),
    st.integers(1, 3),
    st.floats(0, 100, allow_nan=False),
    st.floats(0.1, 20, allow_nan=False),
)


@settings(max_examples=300, deadline=None)
@given(st.lists(turn_strategy, max_size=12), st.integers(0, 5))
def test_properties(turns: list[SpeakerTurn], words: int) -> None:
    utterances = [
        utt(i, t.speaker_label, t.start, t.end, " ".join(["w"] * words))
        for i, t in enumerate(turns)
    ]
    report = analyze(turns, utterances, 50, bucket=7)
    m = report.meeting_stats
    summed = sum(s.speaking_time_seconds for s in report.speakers)
    # each rounded value is off by at most 0.005
    tol = 0.005 * (len(report.speakers) + 2) + 1e-9
    assert summed == pytest.approx(
        m.total_speech_seconds + m.overlap_double_counted_seconds, abs=tol
    )
    assert m.overlap_seconds <= m.overlap_double_counted_seconds + tol
    if utterances:
        shares = sum(s.segment_share_percent for s in report.speakers)
        assert shares == pytest.approx(100, abs=0.005 * len(report.speakers) + 1e-9)
    bucket_total = sum(v for b in report.timeline for v in b.speakers.values())
    assert bucket_total == pytest.approx(summed, abs=0.005 * (len(report.timeline) + 1) * 3 + tol)
    numbers = [
        v for s in report.speakers for v in s.model_dump().values() if isinstance(v, int | float)
    ] + [v for v in m.model_dump().values() if isinstance(v, int | float)]
    assert all(v >= 0 for v in numbers)
    assert 0 <= m.gini_coefficient < 1
