"""Stitch real single-speaker clips into multi-speaker, code-mixed meetings with ground truth.

    uv run python scripts/make_codemixed_meeting.py                    # 8 meetings, seed 7
    uv run python scripts/make_codemixed_meeting.py --meetings 10 --seed 1

Input: the output of ``scripts/prepare_eval_data.py`` (``data/eval``).

* **Code-mixed speakers** come from MUCS 2021 Hindi-English segments. Each MUCS
  recording is one lecturer, so all its segments are the same real voice and one
  meeting speaker gets several turns of genuine intra-sentential Hindi-English.
* **Monolingual speakers** come from FLEURS (English, Hindi, Odia). FLEURS does not
  publish speaker ids, so a FLEURS speaker contributes exactly one clip (one turn);
  otherwise two different voices could share a reference label.

Turns are separated by 0.3-1.0 s pauses; with probability ``--overlap-prob`` a turn
starts 0.4-1.2 s before the previous one ends (mixed additively), so overlap handling
is exercised too.

Output in ``<data>/meetings``, per meeting ``mtg_NN``:

    mtg_NN.wav          16 kHz mono
    mtg_NN.rttm         reference diarization
    mtg_NN.txt          reference transcript (turns in order, for meeting-level CER)
    mtg_NN.lang.json    reference language spans [{start, end, language}]
    mtg_NN.json         everything: speakers, turns (speaker, start, end, language, text, source)

Language labels: FLEURS clips are labelled with their language; MUCS turns are
labelled ``hi`` (Hindi is the matrix language; English words are embedded).
"""

from __future__ import annotations

import argparse
import array
import itertools
import json
import random
import sys
import wave
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

RATE = 16000


@dataclass
class Clip:
    path: Path
    text: str
    language: str
    voice: str  # stable id of the real speaker (MUCS recording, or the FLEURS clip itself)
    source: str


@dataclass
class Turn:
    speaker: str
    start: float
    end: float
    language: str
    text: str
    source: str


def read_samples(path: Path) -> array.array[int]:
    with wave.open(str(path)) as w:
        if (w.getframerate(), w.getnchannels(), w.getsampwidth()) != (RATE, 1, 2):
            raise ValueError(f"{path}: expected 16 kHz mono 16-bit")
        samples = array.array("h")
        samples.frombytes(w.readframes(w.getnframes()))
    if sys.byteorder == "big":
        samples.byteswap()
    return samples


def load_clips(data: Path) -> tuple[dict[str, list[Clip]], dict[str, list[Clip]]]:
    """(MUCS clips grouped by recording, FLEURS clips grouped by language)."""
    mucs: dict[str, list[Clip]] = defaultdict(list)
    for wav in sorted((data / "codemixed").glob("*.wav")):
        rec = wav.stem.rsplit("_", 1)[0]
        text = wav.with_suffix(".txt").read_text("utf-8").strip()
        mucs[rec].append(Clip(wav, text, "hi", f"mucs:{rec}", "mucs_hi_en"))
    fleurs: dict[str, list[Clip]] = defaultdict(list)
    for lang in ("en", "hi", "or"):
        for wav in sorted((data / "asr" / lang).glob("*.wav")):
            text = wav.with_suffix(".txt").read_text("utf-8").strip()
            fleurs[lang].append(Clip(wav, text, lang, f"fleurs:{wav.stem}", "fleurs"))
    return mucs, fleurs


def plan_meeting(
    rng: random.Random,
    mucs: dict[str, list[Clip]],
    fleurs: dict[str, list[Clip]],
    used: set[str],
) -> list[tuple[str, Clip]]:
    """Speaker label -> clip, in speaking order. Clips are never reused across meetings."""
    free_mucs = {r: [c for c in clips if c.path.name not in used] for r, clips in mucs.items()}
    recordings = [r for r, clips in free_mucs.items() if len(clips) >= 2]
    n_mixed = min(len(recordings), rng.choice([1, 2, 2]))
    mixed = rng.sample(recordings, n_mixed)
    mono_langs = ["or", *rng.sample(["en", "hi", "or"], rng.choice([1, 2]))]
    speakers: list[tuple[str, list[Clip]]] = []
    for rec in mixed:
        turns = rng.randint(2, min(4, len(free_mucs[rec])))
        speakers.append((f"mucs:{rec}", rng.sample(free_mucs[rec], turns)))
    for lang in mono_langs:
        free = [c for c in fleurs[lang] if c.path.name not in used]
        if free:
            speakers.append((f"fleurs:{lang}", [rng.choice(free)]))
    for _, clips in speakers:
        used.update(c.path.name for c in clips)
    labels = {i: f"spk{i + 1}" for i in range(len(speakers))}
    # Interleave: repeatedly pick a speaker that still has turns, never the same one twice
    # in a row when another is available.
    queues = {labels[i]: list(clips) for i, (_, clips) in enumerate(speakers)}
    order: list[tuple[str, Clip]] = []
    last = None
    while any(queues.values()):
        options = [s for s, q in queues.items() if q and s != last] or [
            s for s, q in queues.items() if q
        ]
        speaker = rng.choice(options)
        order.append((speaker, queues[speaker].pop(0)))
        last = speaker
    return order


def render(
    order: list[tuple[str, Clip]], rng: random.Random, overlap_prob: float
) -> tuple[array.array[int], list[Turn]]:
    out = array.array("h")
    turns: list[Turn] = []
    cursor = 0.5  # lead-in silence
    for speaker, clip in order:
        samples = read_samples(clip.path)
        duration = len(samples) / RATE
        if turns and rng.random() < overlap_prob:
            previous = turns[-1]
            overlap = min(rng.uniform(0.4, 1.2), 0.4 * (previous.end - previous.start))
            start = previous.end - overlap
        else:
            start = cursor + rng.uniform(0.3, 1.0)
        a = int(start * RATE)
        needed = a + len(samples) - len(out)
        if needed > 0:
            out.extend(array.array("h", bytes(2 * needed)))
        for i, s in enumerate(samples):  # additive mix, clipped to int16
            v = out[a + i] + s
            out[a + i] = 32767 if v > 32767 else -32768 if v < -32768 else v
        end = start + duration
        turns.append(
            Turn(
                speaker,
                round(start, 3),
                round(end, 3),
                clip.language,
                clip.text,
                f"{clip.source}:{clip.path.name}",
            )
        )
        cursor = max(cursor, end)
    out.extend(array.array("h", bytes(2 * RATE)))  # 1 s tail
    return out, turns


def write_meeting(dest: Path, samples: array.array[int], turns: list[Turn]) -> None:
    name = dest.name
    if sys.byteorder == "big":
        samples = array.array("h", samples)
        samples.byteswap()
    with wave.open(str(dest.with_suffix(".wav")), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(samples.tobytes())
    rttm = [
        f"SPEAKER {name} 1 {t.start:.3f} {t.end - t.start:.3f} <NA> <NA> {t.speaker} <NA> <NA>"
        for t in turns
    ]
    dest.with_suffix(".rttm").write_text("\n".join(rttm) + "\n", encoding="utf-8")
    ordered = sorted(turns, key=lambda t: t.start)
    dest.with_suffix(".txt").write_text("\n".join(t.text for t in ordered) + "\n", encoding="utf-8")
    spans = [{"start": t.start, "end": t.end, "language": t.language} for t in ordered]
    dest.with_suffix(".lang.json").write_text(json.dumps(spans, indent=1), encoding="utf-8")
    meta = {
        "meeting": name,
        "duration_seconds": round(len(samples) / RATE, 2),
        "num_speakers": len({t.speaker for t in turns}),
        "languages": sorted({t.language for t in turns}),
        "overlap_turns": sum(1 for a, b in itertools.pairwise(ordered) if b.start < a.end),
        "turns": [asdict(t) for t in ordered],
    }
    dest.with_suffix(".json").write_text(
        json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=Path("data/eval"))
    parser.add_argument("--meetings", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--overlap-prob", type=float, default=0.25)
    args = parser.parse_args(argv)

    mucs, fleurs = load_clips(args.data)
    if not mucs or not fleurs.get("or"):
        print("need data/eval/codemixed and data/eval/asr/or: run prepare_eval_data.py first")
        return 1
    out_dir = args.data / "meetings"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)  # noqa: S311 - reproducible test data, not crypto
    used: set[str] = set()
    for n in range(1, args.meetings + 1):
        order = plan_meeting(rng, mucs, fleurs, used)
        samples, turns = render(order, rng, args.overlap_prob)
        write_meeting(out_dir / f"mtg_{n:02d}", samples, turns)
        speakers = len({t.speaker for t in turns})
        print(
            f"mtg_{n:02d}: {len(samples) / RATE:6.1f} s, {speakers} speakers, "
            f"{len(turns)} turns, languages {sorted({t.language for t in turns})}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
