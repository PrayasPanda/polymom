"""Evaluate decisions and action items against hand-written gold lists.

    uv run python scripts/eval_summary.py                      # LLM_PROVIDER from .env
    LLM_PROVIDER=mock uv run python scripts/eval_summary.py    # offline smoke run
    uv run python scripts/eval_summary.py tests/fixtures/meetings --judge fuzzy --json out.json

Per fixture (``tests/fixtures/meetings/*.json``) it runs the summarizer and reports:

- precision / recall for decisions and action items. A predicted item matches a gold item
  when the judge says they mean the same thing (one-to-one). ``--judge llm`` asks the
  configured LLM with a rubric; ``--judge fuzzy`` uses rapidfuzz token_set_ratio >= 60;
  ``auto`` (default) uses the LLM unless the provider is ``mock``.
- owner accuracy: matched action items whose owner label equals the gold owner exactly.
- grounding pass rate: evidence items that passed verification.
"""

import argparse
import asyncio
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from rapidfuzz import fuzz

from app.core.config import Settings
from app.schemas.transcript import AlignmentStats, SpeakerTranscript, Utterance
from app.services.llm import build_llm_client
from app.services.llm.base import LLMClient
from app.services.summarization.summarizer import Summarizer

FUZZY_MATCH = 60.0
DEFAULT_FIXTURES = Path("tests/fixtures/meetings")

JUDGE_SYSTEM = """You grade meeting-minutes extraction. For each GOLD item, find the PREDICTED
item that expresses the same decision or task (same action and object; wording, language and
script may differ; a prediction that is broader or narrower but clearly about the same thing
counts; a different task does not). Each predicted item may match at most one gold item.
Return JSON only."""


class JudgeMatch(BaseModel):
    gold_index: int
    predicted_index: int | None = Field(description="null when nothing matches")


class JudgeResult(BaseModel):
    matches: list[JudgeMatch]


def load_transcript(fixture: dict[str, Any]) -> SpeakerTranscript:
    utterances = [
        Utterance(
            id=u["id"],
            speaker=u["speaker"],
            start=u["start"],
            end=u["end"],
            duration=u["end"] - u["start"],
            text=u["text"],
            words=[],
            primary_language=u["lang"],
            languages_present=[u["lang"]],
            is_code_mixed=False,
            avg_confidence=None,
            has_overlap=False,
            overlapping_speakers=[],
            alignment_precision="segment",
        )
        for u in fixture["utterances"]
    ]
    return SpeakerTranscript(
        utterances=utterances,
        speakers=sorted({u.speaker for u in utterances}),
        total_duration=max((u.end for u in utterances), default=0.0),
        warnings=[],
        alignment_stats=AlignmentStats(
            total_words=0, percent_assigned=100, percent_unknown=0, percent_segment_level=100
        ),
    )


def fuzzy_match(gold: list[str], predicted: list[str]) -> dict[int, int]:
    """Greedy one-to-one matching by similarity, best pairs first."""
    pairs = sorted(
        (
            (fuzz.token_set_ratio(g.casefold(), p.casefold()), gi, pi)
            for gi, g in enumerate(gold)
            for pi, p in enumerate(predicted)
        ),
        reverse=True,
    )
    matched: dict[int, int] = {}
    for score, gi, pi in pairs:
        if score >= FUZZY_MATCH and gi not in matched and pi not in matched.values():
            matched[gi] = pi
    return matched


async def llm_match(llm: LLMClient, gold: list[str], predicted: list[str]) -> dict[int, int]:
    if not gold or not predicted:
        return {}
    user = (
        "GOLD:\n"
        + "\n".join(f"{i}. {g}" for i, g in enumerate(gold))
        + "\n\nPREDICTED:\n"
        + "\n".join(f"{i}. {p}" for i, p in enumerate(predicted))
    )
    result = await llm.generate_structured(JUDGE_SYSTEM, user, JudgeResult, 0.0, name="judge")
    matched: dict[int, int] = {}
    for m in result.matches:
        valid = m.predicted_index is not None and 0 <= m.predicted_index < len(predicted)
        if valid and 0 <= m.gold_index < len(gold) and m.predicted_index not in matched.values():
            matched[m.gold_index] = m.predicted_index  # type: ignore[assignment]
    return matched


def prf(matched: int, predicted: int, gold: int) -> tuple[float, float]:
    return (matched / predicted if predicted else 1.0), (matched / gold if gold else 1.0)


def load_fixtures(directory: Path) -> list[tuple[str, dict[str, Any]]]:
    return [
        (p.stem, json.loads(p.read_text(encoding="utf-8")))
        for p in sorted(directory.glob("*.json"))
    ]


async def evaluate(
    fixtures: list[tuple[str, dict[str, Any]]], settings: Settings, judge: str
) -> dict[str, Any]:
    llm = build_llm_client(settings)
    use_llm_judge = judge == "llm" or (judge == "auto" and settings.llm_provider != "mock")
    totals = {"dec": [0, 0, 0], "act": [0, 0, 0], "owner": [0, 0], "ev": [0, 0]}
    rows = []
    for name, fixture in fixtures:
        summary = await Summarizer(llm, settings).summarize(
            load_transcript(fixture),
            meeting_date=date.fromisoformat(fixture["date"]),
        )
        gold_dec = [d["text"] for d in fixture["gold"]["decisions"]]
        gold_act = fixture["gold"]["action_items"]
        pred_dec = [d.decision for d in summary.decisions if d.status == "active"]
        pred_act = summary.action_items

        async def match(gold: list[str], predicted: list[str]) -> dict[int, int]:
            if use_llm_judge:
                return await llm_match(llm, gold, predicted)
            return fuzzy_match(gold, predicted)

        dec = await match(gold_dec, pred_dec)
        act = await match([a["task"] for a in gold_act], [a.task for a in pred_act])
        owners = sum(pred_act[pi].owner == gold_act[gi]["owner"] for gi, pi in act.items())
        report = summary.verification_report
        for key, m, p, g in (("dec", dec, pred_dec, gold_dec), ("act", act, pred_act, gold_act)):
            totals[key][0] += len(m)
            totals[key][1] += len(p)
            totals[key][2] += len(g)
        totals["owner"][0] += owners
        totals["owner"][1] += len(act)
        totals["ev"][0] += report.evidence_passed
        totals["ev"][1] += report.evidence_checked
        rows.append(
            {
                "fixture": name,
                "decisions": dict(
                    zip(
                        ("precision", "recall"),
                        prf(len(dec), len(pred_dec), len(gold_dec)),
                        strict=True,
                    )
                ),
                "action_items": dict(
                    zip(
                        ("precision", "recall"),
                        prf(len(act), len(pred_act), len(gold_act)),
                        strict=True,
                    )
                ),
                "owner_accuracy": owners / len(act) if act else None,
                "grounding_pass_rate": report.grounding_pass_rate,
                "injection_flags": len(report.injection_flags),
            }
        )
    d_p, d_r = prf(*totals["dec"])
    a_p, a_r = prf(*totals["act"])
    return {
        "provider": llm.provider,
        "model": llm.model,
        "judge": "llm" if use_llm_judge else "fuzzy",
        "fixtures": rows,
        "overall": {
            "decisions": {"precision": d_p, "recall": d_r},
            "action_items": {"precision": a_p, "recall": a_r},
            "owner_accuracy": totals["owner"][0] / totals["owner"][1]
            if totals["owner"][1]
            else None,
            "grounding_pass_rate": totals["ev"][0] / totals["ev"][1] if totals["ev"][1] else 1.0,
        },
    }


def print_table(result: dict[str, Any]) -> None:
    print(f"provider={result['provider']} model={result['model']} judge={result['judge']}")
    print(
        f"{'fixture':<24}{'dec P':>7}{'dec R':>7}{'act P':>7}{'act R':>7}{'owner':>7}{'ground':>8}"
    )
    for row in [*result["fixtures"], {"fixture": "OVERALL", **result["overall"]}]:
        owner = row["owner_accuracy"]
        print(
            f"{row['fixture']:<24}"
            f"{row['decisions']['precision']:>7.2f}{row['decisions']['recall']:>7.2f}"
            f"{row['action_items']['precision']:>7.2f}{row['action_items']['recall']:>7.2f}"
            f"{'-' if owner is None else f'{owner:.2f}':>7}{row['grounding_pass_rate']:>8.2f}"
        )


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("fixtures", nargs="?", type=Path, default=DEFAULT_FIXTURES)
    parser.add_argument("--judge", choices=["auto", "llm", "fuzzy"], default="auto")
    parser.add_argument("--json", type=Path, help="also write the results here")
    args = parser.parse_args(argv)
    result = asyncio.run(evaluate(load_fixtures(args.fixtures), Settings(), args.judge))
    print_table(result)
    if args.json:
        args.json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


if __name__ == "__main__":
    main(sys.argv[1:])
