"""Download small, license-compatible public evaluation sets (audio is never committed).

    uv run python scripts/prepare_eval_data.py                 # defaults below
    uv run python scripts/prepare_eval_data.py --per-language 50 --ami ES2004a IS1009a

Sources (all verified public and ungated on 2026-09-30):

| Set | Used for | License |
| --- | --- | --- |
| FLEURS test (google/fleurs: en_us, hi_in, or_in) | WER/CER per language, LID | CC-BY-4.0 |
| MUCS 2021 Hindi-English test (OpenSLR 104) | code-mixed CER, code-mixed meetings | CC-BY-SA-4.0 |
| AMI Meeting Corpus, Mix-Headset (+ pyannote AMI-diarization-setup RTTMs) | DER | CC-BY-4.0 |

Not used: Common Voice (download now requires accepting terms on the Mozilla Data
Collective), Kathbath and IndicVoices (gated on Hugging Face), VoxConverse (the
audio archive is ~20 GB; AMI covers multi-speaker diarization at a fraction).

Layout written under ``--out`` (default ``data/eval``, git-ignored)::

    asr/{en,hi,or}/<id>.wav + <id>.txt      FLEURS, 16 kHz mono
    codemixed/<recording>_<n>.wav + .txt    MUCS Hindi-English segments
    ami/<meeting>.wav + <meeting>.rttm      first --ami-minutes of each meeting
    manifest.json                           sources, licenses, counts, sha256

Downloads are cached in ``<out>/.cache``; FLEURS archives are streamed and the
download stops once enough clips are extracted.
"""

from __future__ import annotations

import argparse
import array
import csv
import hashlib
import io
import json
import struct
import sys
import tarfile
import wave
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import IO, Any

import httpx

FLEURS = "https://huggingface.co/datasets/google/fleurs/resolve/main/data/{config}/{path}"
FLEURS_CONFIGS = {"en": "en_us", "hi": "hi_in", "or": "or_in"}
MUCS_TEST = "https://www.openslr.org/resources/104/Hindi-English_test.tar.gz"
AMI_WAV = "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/{m}/audio/{m}.Mix-Headset.wav"
AMI_RTTM = (
    "https://raw.githubusercontent.com/pyannote/AMI-diarization-setup/main/"
    "only_words/rttms/test/{m}.rttm"
)
SOURCES = {
    "fleurs": {
        "url": "https://huggingface.co/datasets/google/fleurs",
        "license": "CC-BY-4.0",
        "citation": "Conneau et al., FLEURS, 2022 (arXiv:2205.12446)",
    },
    "mucs_hi_en": {
        "url": "https://www.openslr.org/104/",
        "license": "CC-BY-SA-4.0",
        "citation": "Diwan et al., MUCS 2021 (Interspeech 2021)",
    },
    "ami": {
        "url": "https://groups.inf.ed.ac.uk/ami/corpus/",
        "license": "CC-BY-4.0",
        "citation": "Carletta et al., The AMI Meeting Corpus, 2005; RTTMs from "
        "github.com/pyannote/AMI-diarization-setup (only_words)",
    },
}
TIMEOUT = httpx.Timeout(60.0, read=300.0)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download(url: str, dest: Path) -> Path:
    """Download once into the cache (resumable by simply re-running)."""
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    with httpx.stream("GET", url, follow_redirects=True, timeout=TIMEOUT) as r:
        r.raise_for_status()
        with part.open("wb") as f:
            for chunk in r.iter_bytes(1 << 20):
                f.write(chunk)
    part.replace(dest)
    return dest


class _HTTPReader(io.RawIOBase):
    """File-like view of a streamed HTTP body, so ``tarfile`` can read it lazily."""

    def __init__(self, response: httpx.Response) -> None:
        self._chunks = response.iter_bytes(1 << 16)
        self._buf = b""

    def readable(self) -> bool:
        return True

    def readinto(self, b: Any) -> int:
        while not self._buf:
            try:
                self._buf = next(self._chunks)
            except StopIteration:
                return 0
        n = min(len(b), len(self._buf))
        b[:n] = self._buf[:n]
        self._buf = self._buf[n:]
        return n


def pcm16_mono(raw: bytes) -> tuple[bytes, int]:
    """(16-bit mono PCM frames, sample rate) from a PCM16 or IEEE-float WAV.

    FLEURS ships 32-bit float WAVs, which the stdlib ``wave`` module cannot read.
    """
    if raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise ValueError("not a RIFF/WAVE file")
    pos, fmt, data, fmt_ext = 12, None, None, b""
    while pos + 8 <= len(raw):
        cid, size = raw[pos : pos + 4], struct.unpack("<I", raw[pos + 4 : pos + 8])[0]
        body = raw[pos + 8 : pos + 8 + size]
        if cid == b"fmt ":
            fmt = struct.unpack("<HHIIHH", body[:16])
            fmt_ext = body[24:]
        elif cid == b"data":
            data = body
        pos += 8 + size + (size & 1)
    if fmt is None or data is None:
        raise ValueError("missing fmt or data chunk")
    tag, channels, rate, _, _, bits = fmt
    if tag == 0xFFFE:  # WAVE_FORMAT_EXTENSIBLE: real tag is the first 2 bytes of SubFormat
        tag = struct.unpack("<H", fmt_ext[:2])[0] if len(fmt_ext) >= 2 else 1
    if tag == 3 and bits == 32:
        floats = array.array("f", data[: len(data) // 4 * 4])
        if sys.byteorder == "big":
            floats.byteswap()
        samples = array.array("h", (int(max(-1.0, min(1.0, v)) * 32767) for v in floats))
    elif tag == 1 and bits == 16:
        samples = array.array("h", data[: len(data) // 2 * 2])
        if sys.byteorder == "big":
            samples.byteswap()
    else:
        raise ValueError(f"unsupported WAV format tag {tag} with {bits} bits")
    if channels > 1:
        samples = array.array("h", samples[::channels])  # first channel
    if sys.byteorder == "big":
        samples.byteswap()
    return samples.tobytes(), rate


def write_wav_16k_mono(raw: bytes, dest: Path) -> float:
    """Write as 16 kHz mono PCM16 (FLEURS and MUCS are 16 kHz already)."""
    frames, rate = pcm16_mono(raw)
    if rate != 16000:
        raise ValueError(f"{dest.name}: expected 16 kHz, got {rate}")
    with wave.open(str(dest), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(frames)
    return len(frames) / 2 / rate


def prepare_fleurs(out: Path, per_language: int) -> dict[str, Any]:
    counts: dict[str, dict[str, float]] = {}
    for lang, config in FLEURS_CONFIGS.items():
        target = out / "asr" / lang
        target.mkdir(parents=True, exist_ok=True)
        existing = sorted(target.glob("*.wav"))
        if len(existing) >= per_language:
            counts[lang] = {"clips": len(existing), "seconds": _total_seconds(existing)}
            continue
        tsv = httpx.get(
            FLEURS.format(config=config, path="test.tsv"), follow_redirects=True, timeout=TIMEOUT
        ).text
        # Columns: id, file, raw transcription, normalized transcription, chars, samples, gender
        rows = {r[1]: r for r in csv.reader(io.StringIO(tsv), delimiter="\t", quoting=3)}
        seen_sentences: set[str] = set()
        taken, seconds = 0, 0.0
        url = FLEURS.format(config=config, path="audio/test.tar.gz")
        with httpx.stream("GET", url, follow_redirects=True, timeout=TIMEOUT) as r:
            r.raise_for_status()
            reader: IO[bytes] = io.BufferedReader(_HTTPReader(r), 1 << 20)
            with tarfile.open(fileobj=reader, mode="r|gz") as tar:
                for member in tar:
                    name = Path(member.name).name
                    row = rows.get(name)
                    # One recording per sentence, so the set covers distinct sentences.
                    if not member.isfile() or row is None or row[0] in seen_sentences:
                        continue
                    f = tar.extractfile(member)
                    if f is None:
                        continue
                    stem = f"fleurs_{lang}_{row[0]}"
                    seconds += write_wav_16k_mono(f.read(), target / f"{stem}.wav")
                    (target / f"{stem}.txt").write_text(row[2].strip() + "\n", encoding="utf-8")
                    seen_sentences.add(row[0])
                    taken += 1
                    if taken >= per_language:
                        break  # stop streaming: only the first part of the archive is read
        counts[lang] = {"clips": taken, "seconds": round(seconds, 1)}
        print(f"FLEURS {lang}: {taken} clips, {seconds / 60:.1f} min")
    return counts


def _total_seconds(wavs: list[Path]) -> float:
    total = 0.0
    for p in wavs:
        with wave.open(str(p)) as w:
            total += w.getnframes() / w.getframerate()
    return round(total, 1)


def prepare_mucs(out: Path, recordings: int, per_recording: int) -> dict[str, Any]:
    """Kaldi-style test set: long per-lecture WAVs + ``segments`` + ``text``."""
    target = out / "codemixed"
    target.mkdir(parents=True, exist_ok=True)
    if any(target.glob("*.wav")):
        wavs = sorted(target.glob("*.wav"))
        return {"segments": len(wavs), "seconds": _total_seconds(wavs)}
    archive = download(MUCS_TEST, out / ".cache" / "Hindi-English_test.tar.gz")
    segments: dict[str, tuple[str, float, float]] = {}
    texts: dict[str, str] = {}
    with tarfile.open(archive, "r:gz") as tar:  # pass 1: metadata only
        for member in tar:
            base = Path(member.name).name
            if base in ("segments", "text") and member.isfile():
                f = tar.extractfile(member)
                if f is None:
                    continue
                for line in f.read().decode("utf-8").splitlines():
                    key, _, rest = line.partition(" ")
                    if base == "segments":
                        rec, start, end = rest.split()
                        segments[key] = (rec, float(start), float(end))
                    else:
                        texts[key] = rest.strip()
    by_recording: dict[str, list[str]] = defaultdict(list)
    for seg, (rec, start, end) in sorted(segments.items(), key=lambda kv: kv[1][1]):
        # 2-15 s segments with a transcript; lectures give one consistent voice per recording.
        if seg in texts and 2.0 <= end - start <= 15.0:
            by_recording[rec].append(seg)
    chosen = sorted(by_recording, key=lambda r: -len(by_recording[r]))[:recordings]
    wanted = {rec: by_recording[rec][:per_recording] for rec in chosen}
    written, seconds = 0, 0.0
    with tarfile.open(archive, "r:gz") as tar:  # pass 2: cut the chosen segments
        for member in tar:
            rec = Path(member.name).stem
            if rec not in wanted or not member.name.endswith(".wav"):
                continue
            f = tar.extractfile(member)
            if f is None:
                continue
            with wave.open(io.BytesIO(f.read())) as w:
                rate = w.getframerate()
                frames = w.readframes(w.getnframes())
                width, channels = w.getsampwidth(), w.getnchannels()
            for i, seg in enumerate(wanted[rec]):
                _, start, end = segments[seg]
                a, b = (int(t * rate) * width * channels for t in (start, end))
                dest = target / f"{rec}_{i:02d}.wav"
                with wave.open(str(dest), "wb") as w:
                    w.setnchannels(channels)
                    w.setsampwidth(width)
                    w.setframerate(rate)
                    w.writeframes(frames[a:b])
                dest.with_suffix(".txt").write_text(texts[seg] + "\n", encoding="utf-8")
                written += 1
                seconds += end - start
    print(f"MUCS hi-en: {written} segments from {len(wanted)} recordings, {seconds / 60:.1f} min")
    return {"segments": written, "recordings": len(wanted), "seconds": round(seconds, 1)}


def prepare_ami(out: Path, meetings: list[str], minutes: float) -> dict[str, Any]:
    target = out / "ami"
    target.mkdir(parents=True, exist_ok=True)
    info: dict[str, Any] = {}
    limit = minutes * 60
    for m in meetings:
        wav = target / f"{m}.wav"
        rttm = target / f"{m}.rttm"
        if not wav.exists():
            full = download(AMI_WAV.format(m=m), out / ".cache" / f"{m}.Mix-Headset.wav")
            with wave.open(str(full)) as src:
                rate = src.getframerate()
                frames = src.readframes(int(min(limit, src.getnframes() / rate) * rate))
                params = src.getparams()
            with wave.open(str(wav), "wb") as dst:
                dst.setparams(params)
                dst.writeframes(frames)
            lines = []
            ref = httpx.get(AMI_RTTM.format(m=m), timeout=TIMEOUT).text
            for line in ref.splitlines():
                parts = line.split()
                start, dur = float(parts[3]), float(parts[4])
                if start >= limit:
                    continue
                parts[4] = f"{min(dur, limit - start):.3f}"
                lines.append(" ".join(parts))
            rttm.write_text("\n".join(lines) + "\n", encoding="utf-8")
        speakers = {line.split()[7] for line in rttm.read_text("utf-8").splitlines() if line}
        info[m] = {"seconds": _total_seconds([wav]), "speakers": len(speakers)}
        print(f"AMI {m}: {info[m]['seconds'] / 60:.1f} min, {len(speakers)} speakers")
    return info


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("data/eval"))
    parser.add_argument("--per-language", type=int, default=30, help="FLEURS clips per language")
    parser.add_argument("--mucs-recordings", type=int, default=6)
    parser.add_argument("--mucs-per-recording", type=int, default=8)
    parser.add_argument("--ami", nargs="*", default=["ES2004a", "IS1009a"])
    parser.add_argument("--ami-minutes", type=float, default=10.0)
    parser.add_argument("--skip", nargs="*", default=[], choices=["fleurs", "mucs", "ami"])
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {"prepared_on": date.today().isoformat(), "sources": SOURCES}
    if "fleurs" not in args.skip:
        manifest["fleurs"] = prepare_fleurs(args.out, args.per_language)
    if "mucs" not in args.skip:
        manifest["mucs_hi_en"] = prepare_mucs(
            args.out, args.mucs_recordings, args.mucs_per_recording
        )
    if "ami" not in args.skip:
        manifest["ami"] = prepare_ami(args.out, args.ami, args.ami_minutes)
    manifest["files"] = {
        str(p.relative_to(args.out).as_posix()): sha256(p)
        for p in sorted(args.out.rglob("*"))
        if p.is_file() and ".cache" not in p.parts and p.name != "manifest.json"
    }
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"wrote {args.out / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
