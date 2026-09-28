import pytest

from app.schemas.audio import AudioWarningCode
from app.services.audio.analysis import AudioAnalysis, parse_analysis, quality_warnings

STDERR = """
  Duration: 00:00:10.00, bitrate: 256 kb/s
[Parsed_silencedetect_0 @ 0x1] silence_start: 0
[Parsed_silencedetect_0 @ 0x1] silence_end: 1.5 | silence_duration: 1.5
[Parsed_silencedetect_0 @ 0x1] silence_start: 4
[Parsed_silencedetect_0 @ 0x1] silence_end: 5 | silence_duration: 1
[Parsed_silencedetect_0 @ 0x1] silence_start: 8.5
[Parsed_ebur128_2 @ 0x2] Summary:

  Integrated loudness:
    I:         -18.4 LUFS
    Threshold: -28.4 LUFS
[Parsed_astats_1 @ 0x3] Channel: 1
[Parsed_astats_1 @ 0x3] Peak level dB: -1.0
[Parsed_astats_1 @ 0x3] RMS level dB: -30.0
[Parsed_astats_1 @ 0x3] Overall
[Parsed_astats_1 @ 0x3] Peak level dB: -2.500000
[Parsed_astats_1 @ 0x3] RMS level dB: -20.250000
"""


def test_parse_analysis_reads_levels_and_silences() -> None:
    analysis = parse_analysis(STDERR)

    assert analysis.duration_seconds == 10.0
    assert analysis.loudness_lufs == -18.4
    assert (analysis.peak_db, analysis.rms_db) == (-2.5, -20.25)  # "Overall", not per-channel
    # The last silence has no silence_end (older ffmpeg at EOF): it runs to the end.
    assert analysis.silences == [(0.0, 1.5), (4.0, 5.0), (8.5, 10.0)]
    assert analysis.silence_ratio == pytest.approx(0.4)
    assert analysis.leading_silence_seconds == 1.5
    assert analysis.trailing_silence_seconds == pytest.approx(1.5)


def test_parse_analysis_handles_digital_silence_and_missing_duration() -> None:
    stderr = """
[Parsed_silencedetect_0 @ 0x1] silence_start: 0
  Integrated loudness:
    I:         -inf LUFS
[Parsed_astats_1 @ 0x3] Overall
[Parsed_astats_1 @ 0x3] Peak level dB: -inf
[Parsed_astats_1 @ 0x3] RMS level dB: -inf
size=N/A time=00:00:03.50 bitrate=N/A
"""
    analysis = parse_analysis(stderr)

    assert analysis.duration_seconds == 3.5
    assert (analysis.loudness_lufs, analysis.rms_db, analysis.peak_db) == (None, None, None)
    assert analysis.silence_ratio == 1.0
    assert analysis.leading_silence_seconds == 3.5
    assert analysis.trailing_silence_seconds == 0.0  # already counted as leading


def test_zero_duration_counts_as_fully_silent() -> None:
    assert AudioAnalysis(0.0, None, None, None).silence_ratio == 1.0


@pytest.mark.parametrize(
    ("analysis", "expected"),
    [
        (AudioAnalysis(60.0, -20.0, -20.0, -3.0), set()),
        (AudioAnalysis(60.0, -50.0, -45.0, -30.0), {AudioWarningCode.LOW_VOLUME}),
        (AudioAnalysis(60.0, None, None, None), {AudioWarningCode.LOW_VOLUME}),
        (AudioAnalysis(60.0, -10.0, -12.0, 0.0), {AudioWarningCode.CLIPPING}),
        (AudioAnalysis(1.5, -20.0, -20.0, -3.0), {AudioWarningCode.TOO_SHORT}),
        (
            AudioAnalysis(10.0, -20.0, -20.0, -3.0, [(0.0, 8.5)]),
            {AudioWarningCode.MOSTLY_SILENT},
        ),
        (AudioAnalysis(10.0, -20.0, -20.0, -3.0, [(0.0, 8.0)]), set()),  # exactly 80%: ok
    ],
)
def test_quality_warnings(analysis: AudioAnalysis, expected: set[AudioWarningCode]) -> None:
    assert {w.code for w in quality_warnings(analysis)} == expected
