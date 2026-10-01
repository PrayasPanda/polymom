"""Benchmark the real pipeline and write docs/evaluation/results.{json,md}.

    uv run python scripts/prepare_eval_data.py && uv run python scripts/make_codemixed_meeting.py
    uv run python scripts/run_benchmark.py                        # everything available
    uv run python scripts/run_benchmark.py --only asr lid         # a subset
    make benchmark-docker                                         # same, in the CUDA image

What is measured (one table in results.md):

* **ASR** WER and CER per language on FLEURS (language hint given), and CER on real
  Hindi-English code-switched speech (MUCS) with *no* hint: spoken LID picks the
  language per region and routes it to the backend, exactly as in production.
* **LID** accuracy: share of speech seconds labelled with the right language, on
  FLEURS clips (clip level) and on the synthetic meetings (region level).
* **Meetings** (synthetic code-mixed + AMI) through the full pipeline (upload ->
  preprocess -> diarize -> LID -> routed ASR -> align -> analytics -> summary): DER
  (collar 0.25 s, overlap scored), speaker-count accuracy, speaker-attributed WER/CER
  (cpWER: best speaker permutation, per-speaker concatenated text), meeting CER.
* **Summaries**: decision and action-item precision/recall and evidence grounding
  rate on the hand-labelled fixtures in tests/fixtures/meetings (scripts/eval_summary.py).
* **Speed**: real-time factor (processing seconds / audio seconds) on the device used.

Nothing is estimated. A metric whose model cannot run here (no HF_TOKEN for the gated
pyannote / IndicConformer models, no LLM for summaries) is written as ``pending`` with
the reason and the command to produce it.
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import os
import platform
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.services.llm import DEFAULT_LLM_MODELS
from scripts import eval_asr, eval_diarization, eval_summary

DATA = Path("data/eval")
OUT = Path("docs/evaluation")
SECTIONS = ("asr", "lid", "meetings", "summary")
COLLAR = 0.25


def pending(reason: str, how: str) -> dict[str, str]:
    return {"status": "pending", "reason": reason, "how_to_run": how}


def hardware() -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        info["torch"] = None
    return info


def model_versions(settings: Any) -> dict[str, str]:
    return {
        "asr_en_hi": f"faster-whisper {settings.whisper_model_size} "
        f"(compute {settings.whisper_compute_type})",
        "asr_or": settings.odia_model_id,
        "lid": settings.lid_model_id if settings.lid_backend == "mms" else settings.lid_backend,
        "diarization": settings.diarization_model,
        "llm": f"{settings.llm_provider}/"
        f"{settings.llm_model or DEFAULT_LLM_MODELS.get(settings.llm_provider, 'default')}",
    }


# --- ASR and LID on clips ------------------------------------------------------------


def lid_accuracy(confusion: dict[str, dict[str, float]]) -> float | None:
    total = sum(sum(row.values()) for row in confusion.values())
    right = sum(row.get(lang, 0.0) for lang, row in confusion.items())
    return right / total if total else None


async def run_asr_and_lid(data: Path, languages: list[str]) -> dict[str, Any]:
    """Hinted WER/CER per language; unhinted (routed) CER + LID confusion."""
    out: dict[str, Any] = {"asr": {}, "lid": {}}
    started = time.perf_counter()
    audio_seconds = 0.0
    for lang in languages:
        samples = eval_asr.discover(data / "asr" / lang, lang)
        audio_seconds += sum(_duration(s.audio) for s in samples)
        pairs = await eval_asr.transcribe_all(samples, "real")
        files, scores = eval_asr.score(pairs)
        out["asr_files"] = out.get("asr_files", []) + [_file_row(f) for f in files]
        out["asr"][lang] = {
            "dataset": "FLEURS test",
            "clips": scores[0].files,
            "wer": scores[0].wer,
            "cer": scores[0].cer,
        }
    # Unhinted, routed: LID decides, per region. Reference language = the folder language.
    confusion: dict[str, dict[str, float]] = {}
    routed_samples = [s for lang in languages for s in eval_asr.discover(data / "asr" / lang, lang)]
    for s in routed_samples:
        s.lang_spans = [(0.0, _duration(s.audio), s.language)]
    audio_seconds += sum(_duration(s.audio) for s in routed_samples)  # second (routed) pass
    await eval_asr.transcribe_all(routed_samples, "real", routed=True, confusion=confusion)
    out["lid"]["fleurs_clips"] = {
        "accuracy": lid_accuracy(confusion),
        "confusion_seconds": confusion,
        "clips": len(routed_samples),
    }
    mixed = eval_asr.discover(data / "codemixed", "hi")
    if mixed:
        for s in mixed:
            s.lang_spans = [(0.0, _duration(s.audio), "hi")]
        audio_seconds += sum(_duration(s.audio) for s in mixed)
        mixed_conf: dict[str, dict[str, float]] = {}
        pairs = await eval_asr.transcribe_all(mixed, "real", routed=True, confusion=mixed_conf)
        files, scores = eval_asr.score([(f, "hi-en", r, h) for f, _, r, h in pairs])
        out["asr_files"] = out.get("asr_files", []) + [_file_row(f) for f in files]
        out["lid"]["mucs_code_mixed"] = {
            "accuracy": lid_accuracy(mixed_conf),
            "confusion_seconds": mixed_conf,
            "clips": len(mixed),
        }
        out["asr"]["hi-en (code-mixed)"] = {
            "dataset": "MUCS 2021 Hindi-English test",
            "clips": scores[0].files,
            "wer": scores[0].wer,
            "cer": scores[0].cer,
            "hint": "none (LID-routed)",
        }
    out["asr_seconds"] = {"audio": audio_seconds, "wall": time.perf_counter() - started}
    return out


def _file_row(f: eval_asr.FileScore) -> dict[str, Any]:
    """Per-clip scores with normalized reference and hypothesis, for error analysis."""
    return {
        "file": Path(f.file).name,
        "language": f.language,
        "wer": f.wer,
        "cer": f.cer,
        "reference": f.reference,
        "hypothesis": f.hypothesis,
    }


def _duration(path: Path) -> float:
    import wave

    with wave.open(str(path)) as w:
        return w.getnframes() / w.getframerate()


# --- Meetings through the full pipeline -----------------------------------------------


def cp_error_rate(reference: dict[str, str], hypothesis: dict[str, str], unit: str) -> float:
    """Concatenated-minimum-permutation error rate (cpWER / cpCER).

    Every reference speaker's text is compared with one hypothesis speaker's text;
    the permutation with the fewest edits wins. Extra or missing speakers are paired
    with empty text, so speaker-count errors are penalised.
    """
    refs = [eval_asr.normalize_for_eval(t) for t in reference.values()]
    hyps = [eval_asr.normalize_for_eval(t) for t in hypothesis.values()]
    size = max(len(refs), len(hyps))
    refs += [""] * (size - len(refs))
    hyps += [""] * (size - len(hyps))
    total = sum(len(r.split()) if unit == "word" else len(r) for r in refs)
    best = None
    # ponytail: brute-force permutations, fine up to ~7 speakers; Hungarian if meetings grow.
    for perm in itertools.permutations(range(size)):
        edits = sum(eval_asr.edit_count(refs[i], hyps[j], unit) for i, j in enumerate(perm))
        best = edits if best is None else min(best, edits)
    return (best or 0.0) / total if total else 0.0


async def run_meetings(data: Path, settings: Any, diarization_real: bool) -> dict[str, Any]:
    from httpx import ASGITransport, AsyncClient

    from app.main import create_app

    meetings = sorted((data / "meetings").glob("mtg_*.wav")) + sorted((data / "ami").glob("*.wav"))
    rows: list[dict[str, Any]] = []
    app = create_app(settings)
    async with (
        app.router.lifespan_context(app),
        AsyncClient(transport=ASGITransport(app=app), base_url="http://bench", timeout=3600) as c,
    ):
        for wav in meetings:
            ami = wav.parent.name == "ami"
            ref_spans = eval_diarization.parse_rttm(wav.with_suffix(".rttm").read_text("utf-8"))
            ref_speakers = {s for _, _, s in ref_spans}
            created = await c.post(
                "/api/v1/meetings",
                files={"file": (wav.name, wav.read_bytes())},
                params={"allow_duplicate": "true"},
            )
            meeting_id = created.json()["meeting_id"]
            started = time.perf_counter()
            await c.post(f"/api/v1/meetings/{meeting_id}/process")  # inline: returns when done
            wall = time.perf_counter() - started
            result = (await c.get(f"/api/v1/meetings/{meeting_id}/result")).json()
            status = result["meeting"]["status"]
            duration = result["meeting"]["duration_seconds"] or _duration(wav)
            row: dict[str, Any] = {
                "meeting": wav.stem,
                "source": "AMI" if ami else "synthetic code-mixed",
                "status": status,
                "duration_seconds": round(duration, 1),
                "ref_speakers": len(ref_speakers),
                "rtf": wall / duration,
                "stage_ms": (result.get("processing") or {}).get("timings_ms", {}),
                "error": result["meeting"]["error"],
            }
            speakers = await c.get(f"/api/v1/meetings/{meeting_id}/speakers")
            if speakers.status_code == 200 and diarization_real:
                body = speakers.json()
                hyp_spans = [(t["start"], t["end"], t["speaker"]) for t in body["turns"]]
                row["der"] = eval_diarization.der(ref_spans, hyp_spans, COLLAR)
                row["hyp_speakers"] = body["num_speakers"]
                row["speaker_count_correct"] = body["num_speakers"] == len(ref_speakers)
            transcript = result.get("transcript")
            if transcript and not ami:
                meta = json.loads(wav.with_suffix(".json").read_text("utf-8"))
                ref_text = " ".join(t["text"] for t in meta["turns"])
                hyp_text = " ".join(u["text"] for u in transcript["utterances"])
                r, h = eval_asr.normalize_for_eval(ref_text), eval_asr.normalize_for_eval(hyp_text)
                row["meeting_cer"] = eval_asr.error_rate([r], [h], "char")
                row["meeting_wer"] = eval_asr.error_rate([r], [h], "word")
                if diarization_real:
                    ref_by: dict[str, str] = {}
                    for t in meta["turns"]:
                        ref_by[t["speaker"]] = f"{ref_by.get(t['speaker'], '')} {t['text']}"
                    hyp_by: dict[str, str] = {}
                    for u in transcript["utterances"]:
                        hyp_by[u["speaker"]] = f"{hyp_by.get(u['speaker'], '')} {u['text']}"
                    row["cpwer"] = cp_error_rate(ref_by, hyp_by, "word")
                    row["cpcer"] = cp_error_rate(ref_by, hyp_by, "char")
                languages = result.get("languages")
                if languages:
                    predicted = [
                        (g["start"], g["end"], g["language"]) for g in languages["regions"]
                    ]
                    reference = [
                        (s["start"], s["end"], s["language"])
                        for s in json.loads(wav.with_suffix(".lang.json").read_text("utf-8"))
                    ]
                    conf = eval_asr.lid_confusion(predicted, reference)
                    row["lid_confusion_seconds"] = conf
                    row["lid_accuracy"] = lid_accuracy(conf)
            rows.append(row)
            print(f"meeting {wav.stem}: {status} rtf={row['rtf']:.3f}", file=sys.stderr)
    return {"meetings": rows}


# --- Report ----------------------------------------------------------------------------


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def aggregate(results: dict[str, Any]) -> list[tuple[str, str, str, str]]:
    """(metric, value, dataset, notes) rows for the one-table summary."""
    rows: list[tuple[str, str, str, str]] = []

    def fmt(v: float | None, pct: bool = True) -> str:
        if v is None:
            return "n/a"
        return f"{100 * v:.1f}%" if pct else f"{v:.3f}"

    def pend(name: str, block: Any, dataset: str) -> bool:
        if isinstance(block, dict) and block.get("status") == "pending":
            rows.append((name, "pending", dataset, block["reason"]))
            return True
        return False

    asr = results.get("asr", {})
    for lang in ("en", "hi", "or"):
        block = asr.get(lang)
        if block is None or pend(f"WER / CER ({lang})", block, "FLEURS test"):
            continue
        rows.append(
            (
                f"WER / CER ({lang})",
                f"{fmt(block['wer'])} / {fmt(block['cer'])}",
                f"FLEURS test, {block['clips']} clips",
                "language hint given",
            )
        )
    mixed = asr.get("hi-en (code-mixed)")
    if mixed and not pend("CER (code-mixed hi-en)", mixed, "MUCS 2021"):
        rows.append(
            (
                "CER (code-mixed hi-en)",
                f"{fmt(mixed['cer'])} (WER {fmt(mixed['wer'])})",
                f"MUCS 2021 test, {mixed['clips']} clips",
                "no hint; LID-routed",
            )
        )
    lid = results.get("lid", {})
    clips = lid.get("fleurs_clips")
    if clips and not pend("LID accuracy (clips)", clips, "FLEURS"):
        rows.append(
            (
                "LID accuracy (clips)",
                fmt(clips["accuracy"]),
                f"FLEURS, {clips['clips']} clips",
                "share of speech seconds",
            )
        )
    mucs = lid.get("mucs_code_mixed")
    if mucs:
        rows.append(
            (
                "LID accuracy (code-mixed clips)",
                fmt(mucs["accuracy"]),
                f"MUCS 2021, {mucs['clips']} clips",
                "reference: hi (matrix language)",
            )
        )
    meetings = results.get("meetings")
    if isinstance(meetings, dict) and "meetings" in meetings:
        ms = meetings["meetings"]
        syn = [m for m in ms if m["source"] != "AMI"]
        ami = [m for m in ms if m["source"] == "AMI"]
        for label, group in (("synthetic", syn), ("AMI", ami)):
            ders = [m["der"] for m in group if "der" in m]
            if ders:
                rows.append(
                    (
                        f"DER ({label})",
                        fmt(mean(ders)),
                        f"{len(ders)} meetings",
                        f"collar {COLLAR}s, overlap scored",
                    )
                )
            counts = [m["speaker_count_correct"] for m in group if "speaker_count_correct" in m]
            if counts:
                rows.append(
                    (
                        f"Speaker-count accuracy ({label})",
                        f"{sum(counts)}/{len(counts)}",
                        f"{len(counts)} meetings",
                        "",
                    )
                )
        if not any("der" in m for m in ms):
            rows.append(
                (
                    "DER, speaker count, cpWER",
                    "pending",
                    "synthetic + AMI",
                    meetings.get("diarization_note", ""),
                )
            )
        cp = [m["cpwer"] for m in syn if "cpwer" in m]
        if cp:
            rows.append(
                (
                    "Speaker-attributed WER / CER (cp)",
                    f"{fmt(mean(cp))} / {fmt(mean([m['cpcer'] for m in syn]))}",
                    f"{len(cp)} synthetic meetings",
                    "best speaker permutation",
                )
            )
        cer = [m["meeting_cer"] for m in syn if "meeting_cer" in m]
        if cer:
            rows.append(
                (
                    "Meeting CER (code-mixed)",
                    fmt(mean(cer)),
                    f"{len(cer)} synthetic meetings",
                    "en+hi+hi-en+or, full pipeline",
                )
            )
        acc = [m["lid_accuracy"] for m in syn if m.get("lid_accuracy") is not None]
        if acc:
            rows.append(
                (
                    "LID accuracy (meetings)",
                    fmt(mean(acc)),
                    f"{len(acc)} synthetic meetings",
                    "region level, share of seconds",
                )
            )
        rtf = [m["rtf"] for m in ms if m["status"].startswith("completed")]
        if rtf:
            rows.append(
                (
                    "Real-time factor (pipeline)",
                    fmt(mean(rtf), pct=False),
                    f"{len(rtf)} meetings",
                    results["hardware"].get("gpu") or "CPU",
                )
            )
    elif meetings is not None:
        pend("Meeting metrics (DER, cpWER, meeting CER)", meetings, "synthetic + AMI")
    secs = results.get("asr_seconds")
    if secs and secs["audio"]:
        rows.append(
            (
                "Real-time factor (ASR+LID clips)",
                fmt(secs["wall"] / secs["audio"], False),
                f"{secs['audio'] / 60:.1f} min audio",
                results["hardware"].get("gpu") or "CPU",
            )
        )
    summary = results.get("summary")
    if summary is not None and not pend("Action items P / R, grounding", summary, "fixtures"):
        o = summary["overall"]
        rows.append(
            (
                "Action items precision / recall",
                f"{fmt(o['action_items']['precision'])} / {fmt(o['action_items']['recall'])}",
                f"{len(summary['fixtures'])} labelled meetings",
                f"{summary['provider']}/{summary['model']}, judge {summary['judge']}",
            )
        )
        rows.append(
            (
                "Decisions precision / recall",
                f"{fmt(o['decisions']['precision'])} / {fmt(o['decisions']['recall'])}",
                f"{len(summary['fixtures'])} labelled meetings",
                "",
            )
        )
        rows.append(
            (
                "Evidence grounding rate",
                fmt(o["grounding_pass_rate"]),
                f"{len(summary['fixtures'])} labelled meetings",
                "quotes verified in transcript",
            )
        )
    return rows


def write_markdown(results: dict[str, Any], path: Path) -> None:
    rows = aggregate(results)
    hw = results["hardware"]
    lines = [
        "# Evaluation results",
        "",
        f"Generated by `scripts/run_benchmark.py` on {results['generated_at']}. "
        "Do not edit by hand; re-run the benchmark instead.",
        "",
        f"- Hardware: {hw.get('gpu') or 'no GPU'} · {hw.get('cpu')} ({hw.get('cpu_count')} threads)"
        f" · {hw.get('platform')}",
        f"- Software: Python {hw.get('python')}, torch {hw.get('torch')} (CUDA {hw.get('cuda')})",
        "- Models: " + "; ".join(f"{k}: `{v}`" for k, v in results["models"].items()),
        "",
        "| Metric | Result | Data | Notes |",
        "| --- | --- | --- | --- |",
    ]
    lines += [f"| {m} | {v} | {d} | {n} |" for m, v, d, n in rows]
    pend = [
        (k, v) for k, v in results.items() if isinstance(v, dict) and v.get("status") == "pending"
    ]
    for key, block in results.get("asr", {}).items():
        if isinstance(block, dict) and block.get("status") == "pending":
            pend.append((f"asr.{key}", block))
    if pend:
        lines += ["", "## Pending measurements", ""]
        lines += [f"- **{k}**: {v['reason']} Run: `{v['how_to_run']}`" for k, v in pend]
    meetings = results.get("meetings", {}).get("meetings", [])
    if meetings:
        lines += [
            "",
            "## Per meeting",
            "",
            "| Meeting | Source | Length | Speakers ref / hyp | DER | cpWER | CER | LID acc "
            "| RTF |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]

        def p(v: Any) -> str:
            return "-" if v is None else f"{100 * v:.1f}%"

        for m in meetings:
            lines.append(
                f"| {m['meeting']} | {m['source']} | {m['duration_seconds']} s | "
                f"{m['ref_speakers']} / {m.get('hyp_speakers', '-')} | {p(m.get('der'))} | "
                f"{p(m.get('cpwer'))} | {p(m.get('meeting_cer'))} | {p(m.get('lid_accuracy'))} | "
                f"{m['rtf']:.3f} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--only", nargs="*", choices=SECTIONS, default=list(SECTIONS))
    args = parser.parse_args(argv)

    from app.core.config import Settings

    settings = Settings()
    has_token = settings.hf_token is not None and bool(settings.hf_token.get_secret_value())
    results: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "hardware": hardware(),
        "models": model_versions(settings),
        "data_manifest": json.loads((args.data / "manifest.json").read_text("utf-8"))
        if (args.data / "manifest.json").exists()
        else None,
    }
    run_hint = "set HF_TOKEN (accept the model terms) and run `make benchmark-docker`"
    languages = ["en", "hi"] + (["or"] if has_token else [])

    if "asr" in args.only or "lid" in args.only:
        asr_lid = asyncio.run(run_asr_and_lid(args.data, languages))
        results.update(asr_lid)
        if not has_token:
            results["asr"]["or"] = pending(
                "Odia ASR (ai4bharat/indic-conformer-600m-multilingual) is gated on Hugging "
                "Face and no HF_TOKEN was available.",
                run_hint,
            )
    if "meetings" in args.only:
        if has_token:
            with tempfile.TemporaryDirectory() as tmp:
                bench = settings.model_copy(
                    update={
                        "storage_dir": Path(tmp),
                        "database_url": None,
                        "pipeline_execution": "inline",
                        "api_key_required": False,
                        "llm_provider": settings.llm_provider,
                        "max_audio_duration_minutes": 240.0,
                    }
                )
                results["meetings"] = asyncio.run(run_meetings(args.data, bench, True))
        else:
            results["meetings"] = pending(
                "Meetings contain Odia and need diarization; pyannote/speaker-diarization-3.1 "
                "and the Odia ASR model are gated and no HF_TOKEN was available.",
                run_hint,
            )
    if "summary" in args.only:
        if settings.llm_provider == "mock":
            results["summary"] = pending(
                "LLM_PROVIDER=mock; summary quality needs a real LLM.",
                "set LLM_PROVIDER (e.g. ollama with --profile local-llm) and re-run",
            )
        else:
            fixtures = eval_summary.load_fixtures(eval_summary.DEFAULT_FIXTURES)
            try:
                results["summary"] = asyncio.run(eval_summary.evaluate(fixtures, settings, "auto"))
            except Exception as exc:  # keep the speech results, report the gap
                results["summary"] = pending(
                    f"The LLM call failed ({type(exc).__name__}: {exc}).",
                    "free the GPU, then: make benchmark-docker ARGS='--only summary'",
                )

    args.out.mkdir(parents=True, exist_ok=True)
    previous = args.out / "results.json"
    if previous.exists() and set(args.only) != set(SECTIONS):
        # A partial run (--only) keeps the sections it did not re-measure.
        old = json.loads(previous.read_text("utf-8"))
        for key, value in old.items():
            if key not in results:
                results[key] = value
        results.setdefault("partial_runs", old.get("partial_runs", []))
        results["partial_runs"].append(
            {"sections": args.only, "generated_at": results["generated_at"]}
        )
    (args.out / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    write_markdown(results, args.out / "results.md")
    print((args.out / "results.md").read_text("utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
