import importlib.util
import json
from pathlib import Path

import pytest

from scripts import eval_diarization as ev

RTTM = """SPEAKER m 1 0.00 5.00 <NA> <NA> alice <NA> <NA>
SPEAKER m 1 5.00 5.00 <NA> <NA> bob <NA> <NA>
; comment line
"""
WORDS = "\n".join(
    "\t".join(row)
    for row in [
        ("0.0", "1.0", "alice", "hello"),
        ("1.0", "2.0", "alice", "all"),
        ("5.0", "6.0", "bob", "yes"),
        ("9.0", "9.5", "bob", "ok"),
        ("30", "31", "bob", "lost"),
    ]
)
TRANSCRIPT = {
    "utterances": [
        {"start": 0, "end": 2, "speaker": "Person 1", "words": [
            {"start": 0.0, "end": 1.0, "speaker": "Person 1", "text": "hello"},
            {"start": 1.0, "end": 2.0, "speaker": "Person 1", "text": "all"},
        ]},
        {"start": 5, "end": 10, "speaker": "Person 2", "words": [
            {"start": 5.1, "end": 6.0, "speaker": "Person 2", "text": "yes"},
            {"start": 9.0, "end": 9.5, "speaker": "Person 1", "text": "ok"},
        ]},
    ]
}  # fmt: skip


def test_parsers() -> None:
    assert ev.parse_rttm(RTTM) == [(0.0, 5.0, "alice"), (5.0, 10.0, "bob")]
    assert ev.parse_words(WORDS)[0] == (0.0, 1.0, "alice", "hello")


def test_wder_with_speaker_mapping() -> None:
    _, words = ev.hypothesis_from_transcript(TRANSCRIPT)

    value, matched = ev.wder(ev.parse_words(WORDS), words)

    # "lost" matches nothing; of 4 matched words, "ok" is attributed to Person 1 (= alice).
    assert (value, matched) == (0.25, 4)
    assert ev.map_speakers([("alice", "Person 1"), ("alice", "Person 1"), ("bob", "Person 2")]) == {
        "Person 1": "alice",
        "Person 2": "bob",
    }


def test_wder_without_matches() -> None:
    assert ev.wder([(0, 1, "a", "x")], []) == (0.0, 0)


def test_evaluate_and_report(tmp_path: Path) -> None:
    (tmp_path / "m.rttm").write_text(RTTM, encoding="utf-8")
    (tmp_path / "m.json").write_text(json.dumps(TRANSCRIPT), encoding="utf-8")
    (tmp_path / "m.words.tsv").write_text(WORDS, encoding="utf-8")
    (tmp_path / "orphan.rttm").write_text(RTTM, encoding="utf-8")
    out = tmp_path / "r.json"

    assert ev.main([str(tmp_path), "--json", str(out)]) == 0

    (score,) = json.loads(out.read_text(encoding="utf-8"))
    assert (score["meeting"], score["wder"], score["matched_words"]) == ("m", 0.25, 4)
    assert "m " in ev.report([ev.MeetingScore("m", None, None, 0)])
    assert ev.main([str(tmp_path / "missing")]) == 1


HAS_METRICS = importlib.util.find_spec("pyannote") is not None


@pytest.mark.skipif(not HAS_METRICS, reason="needs the ml extra (pyannote.metrics)")
def test_der() -> None:
    spans, _ = ev.hypothesis_from_transcript(TRANSCRIPT)

    assert 0 <= ev.der(ev.parse_rttm(RTTM), spans, collar=0.0) <= 1
