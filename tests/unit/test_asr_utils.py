from pathlib import Path

import pytest

from app.schemas.asr import TranscriptSegment
from app.services.asr.segmentation import speech_regions
from app.utils.subtitles import format_timestamp, to_srt, to_text
from scripts import eval_asr


def seg(start: float, end: float, text: str, language: str | None = "hi") -> TranscriptSegment:
    return TranscriptSegment(
        id=0, start=start, end=end, text=text, language=language, words=[],
        avg_confidence=0.9, backend="t",
    )  # fmt: skip


# --- SRT / text -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [(0, "00:00:00,000"), (1.5, "00:00:01,500"), (3661.0456, "01:01:01,046"), (-2, "00:00:00,000")],
)
def test_format_timestamp(seconds: float, expected: str) -> None:
    assert format_timestamp(seconds) == expected


def test_to_srt_numbers_cues_and_preserves_script() -> None:
    srt = to_srt([seg(4.0, 6.25, "ଆଜି ବଜେଟ୍", "or"), seg(0.5, 3.0, "नमस्ते सभी को।")])

    assert srt == (
        "1\n00:00:00,500 --> 00:00:03,000\nनमस्ते सभी को।\n\n"
        "2\n00:00:04,000 --> 00:00:06,250\nଆଜି ବଜେଟ୍\n"
    )


def test_to_srt_zero_length_cue_gets_positive_duration() -> None:
    assert "00:00:01,000 --> 00:00:01,001" in to_srt([seg(1.0, 1.0, "x")])


def test_to_srt_and_text_empty() -> None:
    assert to_srt([]) == ""
    assert to_text([]) == ""


def test_to_text() -> None:
    assert to_text([seg(61.2, 62, "Hello", "en"), seg(0, 1, "?", None)]) == (
        "[00:00:00.000] ?\n[00:01:01.200] (en) Hello\n"
    )


# --- speech regions ------------------------------------------------------------------


def test_speech_regions_bridges_short_pauses_and_drops_blips() -> None:
    frames = [-60.0] * 5 + [-20.0] * 20 + [-60.0] * 5 + [-20.0] * 20 + [-60.0] * 30 + [-20.0] * 3
    # 30 ms frames: speech 0.15-0.75, short pause (0.15 s) bridged, speech to 1.5, silence,
    # then a 90 ms blip that is too short to keep.
    assert speech_regions(frames) == [(0.15, 1.5)]


def test_speech_regions_splits_long_regions() -> None:
    regions = speech_regions([-10.0] * 1500, max_region=20.0)  # 45 s of speech

    assert regions == [(0.0, 15.0), (15.0, 30.0), (30.0, 45.0)]


def test_speech_regions_empty_and_trailing_speech() -> None:
    assert speech_regions([]) == []
    assert speech_regions([-60.0] * 10 + [-20.0] * 20) == [(0.3, 0.9)]


# --- eval script ----------------------------------------------------------------------


def test_normalize_for_eval() -> None:
    assert eval_asr.normalize_for_eval("Hello, WORLD!  नमस्ते।  ଭାଷା॥") == "hello world नमस्ते ଭାଷା"


def test_score_per_language() -> None:
    files, languages = eval_asr.score(
        [
            ("a.wav", "en", "the cat sat", "the cat sat"),
            ("b.wav", "en", "the dog ran", "the dog"),
            ("c.wav", "hi", "नमस्ते दोस्त", "नमस्ते दोस्तो"),
            ("d.wav", "hi", "।", "anything"),  # empty reference after normalization: skipped
        ]
    )

    by_lang = {s.language: s for s in languages}
    assert by_lang["en"].files == 2
    assert by_lang["en"].wer == pytest.approx(1 / 6)
    assert by_lang["hi"].cer == pytest.approx(1 / 12)  # one inserted vowel sign
    assert by_lang["hi"].wer == 0.5
    assert [f.file for f in files] == ["a.wav", "b.wav", "c.wav"]
    assert "en" in eval_asr.report(files, languages)


def test_discover(tmp_path: Path) -> None:
    (tmp_path / "hi").mkdir()
    (tmp_path / "hi" / "a.wav").write_bytes(b"x")
    (tmp_path / "hi" / "a.txt").write_text(" नमस्ते \n", encoding="utf-8")
    (tmp_path / "b.mp3").write_bytes(b"x")  # no reference: skipped
    (tmp_path / "c.flac").write_bytes(b"x")
    (tmp_path / "c.txt").write_text("hi", encoding="utf-8")

    samples = eval_asr.discover(tmp_path)

    assert [(s.audio.name, s.language, s.reference) for s in samples] == [
        ("c.flac", None, "hi"),
        ("a.wav", "hi", "नमस्ते"),
    ]
    assert [s.language for s in eval_asr.discover(tmp_path, "or")] == ["or", "or"]


def test_main_without_samples(tmp_path: Path) -> None:
    assert eval_asr.main([str(tmp_path)]) == 1


def test_lid_confusion_and_report() -> None:
    predicted = [(0.0, 6.0, "en"), (6.0, 10.0, "or")]
    reference = [(0.0, 5.0, "en"), (5.0, 10.0, "hi")]

    matrix = eval_asr.lid_confusion(predicted, reference)
    total: dict[str, dict[str, float]] = {}
    eval_asr.merge_confusion(total, matrix)
    eval_asr.merge_confusion(total, matrix)

    assert matrix == {"en": {"en": 5.0}, "hi": {"en": 1.0, "or": 4.0}}
    assert total["hi"]["or"] == 8.0
    report = eval_asr.confusion_report(matrix)
    assert report.startswith("LID accuracy 0.500 over 10.0s")
    assert eval_asr.confusion_report({}) == "LID: no reference languages"
