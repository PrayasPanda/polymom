"""Evaluate ASR accuracy (WER and CER) per language.

Layout: audio files with a same-named ``.txt`` reference next to them. The
language comes from the parent folder name (``en/``, ``hi/``, ``or/``) unless
``--language`` is given::

    data/
      en/meeting1.wav   en/meeting1.txt
      hi/clip.mp3       hi/clip.txt
      or/sample.wav     or/sample.txt

Usage::

    uv run python scripts/eval_asr.py data/                     # real models
    uv run python scripts/eval_asr.py data/ --backend mock      # plumbing check
    uv run python scripts/eval_asr.py data/ --json report.json

Text is NFC-normalized, case-folded and stripped of punctuation (including the
danda) before scoring. CER is the more meaningful metric for Hindi and Odia,
where word segmentation and spelling variants inflate WER.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
import unicodedata
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".mp4", ".webm", ".mkv"}
LANGUAGES = {"en", "hi", "or"}


def normalize_for_eval(text: str) -> str:
    """NFC, case-fold, punctuation to spaces (Unicode letters/marks/digits kept)."""
    text = unicodedata.normalize("NFC", text).casefold()
    kept = "".join(c if unicodedata.category(c)[0] in "LMN" else " " for c in text)
    return " ".join(kept.split())


@dataclass
class Sample:
    audio: Path
    reference: str
    language: str | None


@dataclass
class FileScore:
    file: str
    language: str
    wer: float
    cer: float
    reference: str
    hypothesis: str


@dataclass
class LanguageScore:
    language: str
    files: int
    wer: float
    cer: float


def discover(data_dir: Path, language: str | None = None) -> list[Sample]:
    samples = []
    for audio in sorted(p for p in data_dir.rglob("*") if p.suffix.lower() in AUDIO_EXTENSIONS):
        ref = audio.with_suffix(".txt")
        if not ref.exists():
            print(f"skip {audio}: no reference {ref.name}", file=sys.stderr)
            continue
        lang = language or (audio.parent.name if audio.parent.name in LANGUAGES else None)
        samples.append(Sample(audio, ref.read_text(encoding="utf-8").strip(), lang))
    return samples


def score(pairs: list[tuple[str, str, str, str]]) -> tuple[list[FileScore], list[LanguageScore]]:
    """``pairs``: (file, language, reference, hypothesis). Corpus-level scores per language."""
    import jiwer

    files: list[FileScore] = []
    grouped: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for name, lang, ref, hyp in pairs:
        r, h = normalize_for_eval(ref), normalize_for_eval(hyp)
        if not r:
            continue
        files.append(FileScore(name, lang, jiwer.wer(r, h or " "), jiwer.cer(r, h or " "), r, h))
        grouped[lang].append((r, h or " "))
    languages = [
        LanguageScore(
            language=lang,
            files=len(items),
            wer=jiwer.wer([r for r, _ in items], [h for _, h in items]),
            cer=jiwer.cer([r for r, _ in items], [h for _, h in items]),
        )
        for lang, items in sorted(grouped.items())
    ]
    return files, languages


async def transcribe_all(samples: list[Sample], backend: str) -> list[tuple[str, str, str, str]]:
    from app.core.config import Settings
    from app.services.asr.service import TranscriptionService, build_router
    from app.services.audio.preprocessor import AudioPreprocessor

    results = []
    with tempfile.TemporaryDirectory() as tmp:
        settings = Settings(storage_dir=Path(tmp), asr_backend=backend)  # also reads .env
        preprocessor = AudioPreprocessor(settings)
        service = TranscriptionService(build_router(settings), settings)
        for sample in samples:
            meeting_id = uuid.uuid4()
            processed = await preprocessor.process(meeting_id, sample.audio)
            hints = [sample.language] if sample.language else []
            asr = await service.transcribe(meeting_id, processed.processed_path, hints)
            hypothesis = " ".join(seg.text for seg in asr.segments)
            detected = sample.language or (
                asr.detected_languages[0].language if asr.detected_languages else "unknown"
            )
            results.append((str(sample.audio), detected, sample.reference, hypothesis))
            print(f"done {sample.audio.name} [{detected}]", file=sys.stderr)
    return results


def report(files: list[FileScore], languages: list[LanguageScore]) -> str:
    lines = [f"{'language':<10}{'files':>6}{'WER':>9}{'CER':>9}"]
    lines += [f"{s.language:<10}{s.files:>6}{s.wer:>9.3f}{s.cer:>9.3f}" for s in languages]
    lines.append("")
    lines += [
        f"{Path(f.file).name:<40} {f.language:<4} WER {f.wer:.3f}  CER {f.cer:.3f}" for f in files
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--backend", choices=["real", "mock"], default="real")
    parser.add_argument("--language", choices=sorted(LANGUAGES), help="force one language")
    parser.add_argument("--json", type=Path, help="write the full report as JSON")
    args = parser.parse_args(argv)

    samples = discover(args.data_dir, args.language)
    if not samples:
        print("no audio files with .txt references found", file=sys.stderr)
        return 1
    files, languages = score(asyncio.run(transcribe_all(samples, args.backend)))
    print(report(files, languages))
    if args.json:
        payload = {"languages": [asdict(s) for s in languages], "files": [asdict(f) for f in files]}
        args.json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
