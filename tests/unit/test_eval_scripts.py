"""Evaluation helpers: scoring, float-WAV conversion and synthetic meeting rendering."""

import array
import io
import random
import struct
import wave
from pathlib import Path

import pytest

from scripts import eval_asr, make_codemixed_meeting, prepare_eval_data, run_benchmark


def test_error_rate_matches_corpus_definition() -> None:
    assert eval_asr.error_rate(["a b c d"], ["a x c"], "word") == pytest.approx(2 / 4)
    assert eval_asr.error_rate(["abcd", "ef"], ["abed", "ef"], "char") == pytest.approx(1 / 6)
    assert eval_asr.error_rate([""], [""], "word") == 0.0


def test_cp_error_rate_is_permutation_invariant_and_penalises_speakers() -> None:
    ref = {"A": "hello there friend", "B": "नमस्ते दोस्त"}
    assert (
        run_benchmark.cp_error_rate(ref, {"x": "नमस्ते दोस्त", "y": "hello there friend"}, "word") == 0
    )
    # One speaker missing: all of B's words are deletions.
    merged = run_benchmark.cp_error_rate(ref, {"x": "hello there friend"}, "word")
    assert merged == pytest.approx(2 / 5)
    # An extra, spurious speaker costs insertions.
    extra = {"x": "hello there friend", "y": "नमस्ते दोस्त", "z": "uh"}
    assert run_benchmark.cp_error_rate(ref, extra, "word") == pytest.approx(1 / 5)


def test_lid_accuracy_from_confusion() -> None:
    assert run_benchmark.lid_accuracy({"hi": {"hi": 3.0, "en": 1.0}, "or": {"or": 4.0}}) == 0.875
    assert run_benchmark.lid_accuracy({}) is None


def _float_wav(values: list[float], rate: int = 16000) -> bytes:
    data = struct.pack(f"<{len(values)}f", *values)
    fmt = struct.pack("<HHIIHH", 3, 1, rate, rate * 4, 4, 32)
    body = b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt
    body += b"data" + struct.pack("<I", len(data)) + data
    return b"RIFF" + struct.pack("<I", len(body)) + body


def test_float_wav_is_converted_to_pcm16(tmp_path: Path) -> None:
    raw = _float_wav([0.0, 0.5, -1.0, 2.0])  # 2.0 is clipped
    frames, rate = prepare_eval_data.pcm16_mono(raw)
    assert rate == 16000
    assert list(array.array("h", frames)) == [0, 16383, -32767, 32767]
    dest = tmp_path / "x.wav"
    assert prepare_eval_data.write_wav_16k_mono(raw, dest) == pytest.approx(4 / 16000)
    with wave.open(str(dest)) as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, 16000)
    with pytest.raises(ValueError, match="RIFF"):
        prepare_eval_data.pcm16_mono(b"nope")


def _clip(path: Path, seconds: float, value: int) -> Path:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(array.array("h", [value] * int(seconds * 16000)).tobytes())
    path.write_bytes(buf.getvalue())
    return path


def test_render_meeting_writes_consistent_ground_truth(tmp_path: Path) -> None:
    clip = make_codemixed_meeting.Clip
    order = [
        ("spk1", clip(_clip(tmp_path / "a.wav", 2.0, 100), "namaste", "hi", "v1", "mucs")),
        ("spk2", clip(_clip(tmp_path / "b.wav", 1.0, 200), "hello", "en", "v2", "fleurs")),
        ("spk1", clip(_clip(tmp_path / "c.wav", 1.5, 300), "theek hai", "hi", "v1", "mucs")),
    ]
    samples, turns = make_codemixed_meeting.render(order, random.Random(0), overlap_prob=1.0)  # noqa: S311
    assert [t.speaker for t in turns] == ["spk1", "spk2", "spk1"]
    assert turns[1].start < turns[0].end  # overlap forced
    assert len(samples) >= int(turns[-1].end * 16000)
    overlap_sample = int((turns[1].start + 0.01) * 16000)
    assert samples[overlap_sample] == 300  # additive mix of 100 + 200

    dest = tmp_path / "mtg_01"
    make_codemixed_meeting.write_meeting(dest, samples, turns)
    rttm = dest.with_suffix(".rttm").read_text("utf-8").splitlines()
    assert len(rttm) == 3
    assert rttm[0].split()[7] == "spk1"
    assert dest.with_suffix(".txt").read_text("utf-8").split("\n")[0] == "namaste"
    assert "overlap_turns" in dest.with_suffix(".json").read_text("utf-8")
