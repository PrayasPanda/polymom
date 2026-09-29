"""Evaluate diarization (DER) and word-level speaker attribution (WDER).

For each ``<stem>.rttm`` reference in the data folder, the system output is
``<stem>.json`` (the body of ``GET /meetings/{id}/transcript``). An optional
``<stem>.words.tsv`` reference transcript enables WDER; one word per line:

    start<TAB>end<TAB>speaker<TAB>word

Usage::

    uv run python scripts/eval_diarization.py data/
    uv run python scripts/eval_diarization.py data/ --collar 0.25 --json report.json

* **DER** (diarization error rate: missed speech + false alarm + speaker
  confusion) uses ``pyannote.metrics`` with its optimal speaker mapping; it needs
  the ``ml`` extra (pyannote.audio pulls pyannote.metrics in).
* **WDER** (word diarization error rate) is the share of reference words, among
  those matched to a system word by time, whose system speaker is wrong after
  the best one-to-one speaker mapping. It isolates attribution from recognition
  errors, so it is comparable across languages.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

Span = tuple[float, float, str]  # start, end, speaker
TimedWord = tuple[float, float, str, str]  # start, end, speaker, text

MATCH_TOLERANCE_SECONDS = 0.5


def parse_rttm(text: str) -> list[Span]:
    """``SPEAKER <file> <chan> <start> <duration> <NA> <NA> <speaker> <NA> <NA>`` lines."""
    spans = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 8 and parts[0] == "SPEAKER":
            start, duration = float(parts[3]), float(parts[4])
            spans.append((start, start + duration, parts[7]))
    return spans


def parse_words(text: str) -> list[TimedWord]:
    words = []
    for line in text.splitlines():
        parts = line.split("\t")
        if len(parts) >= 4 and line.strip():
            words.append((float(parts[0]), float(parts[1]), parts[2], parts[3]))
    return words


def hypothesis_from_transcript(body: dict[str, Any]) -> tuple[list[Span], list[TimedWord]]:
    """Utterance spans and attributed words from a /transcript JSON body."""
    spans = [(u["start"], u["end"], u["speaker"]) for u in body["utterances"]]
    words = [
        (w["start"], w["end"], w["speaker"], w["text"])
        for u in body["utterances"]
        for w in u["words"]
    ]
    return spans, words


def match_words(reference: list[TimedWord], system: list[TimedWord]) -> list[tuple[str, str]]:
    """(reference speaker, system speaker) for each reference word matched in time.

    Match = the system word with the largest overlap, else the nearest midpoint
    within ``MATCH_TOLERANCE_SECONDS``. Unmatched reference words are skipped.
    """
    pairs = []
    for r_start, r_end, r_spk, _ in reference:
        best: tuple[float, float, str] | None = None  # (-overlap, distance, speaker)
        r_mid = (r_start + r_end) / 2
        for s_start, s_end, s_spk, _ in system:
            overlap = min(r_end, s_end) - max(r_start, s_start)
            distance = abs(r_mid - (s_start + s_end) / 2)
            if overlap <= 0 and distance > MATCH_TOLERANCE_SECONDS:
                continue
            key = (-max(overlap, 0.0), distance, s_spk)
            if best is None or key < best:
                best = key
        if best is not None:
            pairs.append((r_spk, best[2]))
    return pairs


def map_speakers(pairs: list[tuple[str, str]]) -> dict[str, str]:
    """One-to-one system -> reference mapping, greedily by co-occurrence count.

    # ponytail: greedy, not Hungarian; optimal for the usual few speakers with a
    # clear majority, swap in scipy.optimize.linear_sum_assignment if needed.
    """
    counts = Counter((sys_spk, ref_spk) for ref_spk, sys_spk in pairs)
    mapping: dict[str, str] = {}
    used: set[str] = set()
    for (sys_spk, ref_spk), _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        if sys_spk not in mapping and ref_spk not in used:
            mapping[sys_spk] = ref_spk
            used.add(ref_spk)
    return mapping


def wder(reference: list[TimedWord], system: list[TimedWord]) -> tuple[float, int]:
    """(WDER, number of matched reference words)."""
    pairs = match_words(reference, system)
    if not pairs:
        return 0.0, 0
    mapping = map_speakers(pairs)
    wrong = sum(mapping.get(sys_spk) != ref_spk for ref_spk, sys_spk in pairs)
    return wrong / len(pairs), len(pairs)


def der(reference: list[Span], system: list[Span], collar: float) -> float:
    from pyannote.core import Annotation, Segment
    from pyannote.metrics.diarization import DiarizationErrorRate

    def annotation(spans: list[Span]) -> Any:
        ann = Annotation()
        for i, (start, end, speaker) in enumerate(spans):
            if end > start:
                ann[Segment(start, end), i] = speaker
        return ann

    metric = DiarizationErrorRate(collar=collar, skip_overlap=False)
    return float(metric(annotation(reference), annotation(system)))


@dataclass
class MeetingScore:
    meeting: str
    der: float | None
    wder: float | None
    matched_words: int


def evaluate(data_dir: Path, collar: float) -> list[MeetingScore]:
    scores = []
    for rttm in sorted(data_dir.rglob("*.rttm")):
        hyp_file = rttm.with_suffix(".json")
        if not hyp_file.exists():
            print(f"skip {rttm.name}: no system output {hyp_file.name}", file=sys.stderr)
            continue
        spans, words = hypothesis_from_transcript(json.loads(hyp_file.read_text("utf-8")))
        try:
            der_value: float | None = der(parse_rttm(rttm.read_text("utf-8")), spans, collar)
        except ImportError:
            print("pyannote.metrics missing: install `uv sync --extra ml` for DER", file=sys.stderr)
            der_value = None
        words_file = rttm.with_suffix(".words.tsv")
        wder_value, matched = (
            wder(parse_words(words_file.read_text("utf-8")), words)
            if words_file.exists()
            else (None, 0)
        )
        scores.append(MeetingScore(rttm.stem, der_value, wder_value, matched))
    return scores


def report(scores: list[MeetingScore]) -> str:
    def fmt(value: float | None) -> str:
        return f"{value:.3f}" if value is not None else "n/a"

    lines = [f"{'meeting':<30}{'DER':>8}{'WDER':>8}{'words':>8}"]
    lines += [f"{s.meeting:<30}{fmt(s.der):>8}{fmt(s.wder):>8}{s.matched_words:>8}" for s in scores]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--collar", type=float, default=0.25, help="DER forgiveness collar (s)")
    parser.add_argument("--json", type=Path, help="write the report as JSON")
    args = parser.parse_args(argv)

    scores = evaluate(args.data_dir, args.collar)
    if not scores:
        print("no <stem>.rttm + <stem>.json pairs found", file=sys.stderr)
        return 1
    print(report(scores))
    if args.json:
        args.json.write_text(json.dumps([asdict(s) for s in scores], indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
