"""Checks every summary item against the transcript before it is shown.

For each piece of evidence:

1. every utterance id exists;
2. the quoted speaker spoke one of those utterances;
3. timestamps, if the model gave any, match the utterances (within 1 s), and are
   then filled in from the transcript;
4. the quote fuzzy-matches the utterance text (rapidfuzz ``partial_ratio`` >=
   ``EVIDENCE_MATCH_THRESHOLD``, after NFC normalization and case folding).

Failing evidence is removed. An item with no evidence left is dropped; an item
that lost some evidence has its confidence lowered one level. Owners, deciders
and askers that are not real speaker labels are reset ("Unassigned", "group",
or the quoted speaker). Everything is recorded in the :class:`VerificationReport`.
"""

import unicodedata
from collections.abc import Sequence
from datetime import date
from typing import TypeVar

from rapidfuzz import fuzz

from app.schemas.summary import (
    GROUP,
    UNASSIGNED,
    ActionItem,
    ChunkExtraction,
    Confidence,
    Decision,
    Evidence,
    InjectionFlag,
    KeyPoint,
    OpenQuestion,
    VerificationIssue,
    VerificationReport,
)
from app.schemas.transcript import Utterance

TIMESTAMP_TOLERANCE_SECONDS = 1.0
_DOWNGRADE: dict[Confidence, Confidence] = {"high": "medium", "medium": "low", "low": "low"}

Item = TypeVar("Item", KeyPoint, Decision, ActionItem, OpenQuestion)


def normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


class Verifier:
    def __init__(self, utterances: Sequence[Utterance], threshold: float) -> None:
        self.by_id = {u.id: u for u in utterances}
        self.speakers = {u.speaker for u in utterances}
        self.threshold = threshold
        self.issues: list[VerificationIssue] = []
        self.checked = self.passed = self.dropped = self.downgraded = 0
        self.evidence_checked = self.evidence_passed = 0

    def check_evidence(self, ev: Evidence) -> str | None:
        """Reason the evidence fails, or ``None`` (and fill in its timestamps)."""
        missing = [i for i in ev.utterance_ids if i not in self.by_id]
        if missing:
            return f"unknown utterance id(s) {missing}"
        refs = [self.by_id[i] for i in ev.utterance_ids]
        if ev.speaker not in {u.speaker for u in refs}:
            return f"speaker {ev.speaker!r} did not say u{ev.utterance_ids}"
        start, end = min(u.start for u in refs), max(u.end for u in refs)
        for given, actual in ((ev.start, start), (ev.end, end)):
            if given is not None and abs(given - actual) > TIMESTAMP_TOLERANCE_SECONDS:
                return f"timestamp {given} does not match the transcript ({actual})"
        quote = normalize(ev.quote)
        text = normalize(" ".join(u.text for u in refs))
        score = fuzz.partial_ratio(quote, text) if quote else 0.0
        if score < self.threshold:
            return f"quote does not match the utterance text (score {score:.0f})"
        ev.start, ev.end = start, end
        return None

    def check_item(self, kind: str, label: str, item: Item) -> Item | None:
        self.checked += 1
        kept: list[Evidence] = []
        reasons: list[str] = []
        for ev in item.evidence:
            self.evidence_checked += 1
            reason = self.check_evidence(ev)
            if reason is None:
                self.evidence_passed += 1
                kept.append(ev)
            else:
                reasons.append(reason)
        if not kept:
            self.dropped += 1
            self._issue(kind, label, "dropped", "; ".join(reasons) or "no evidence")
            return None
        item.evidence = kept
        changed = False
        if reasons:
            changed = True
            self._issue(kind, label, "evidence_removed", "; ".join(reasons))
        if isinstance(item, ActionItem) and item.owner not in self.speakers | {UNASSIGNED}:
            changed = True
            self._issue(kind, label, "owner_reset", f"owner {item.owner!r} is not a speaker")
            item.owner = UNASSIGNED
        if isinstance(item, Decision) and item.made_by not in self.speakers | {GROUP}:
            changed = True
            self._issue(kind, label, "owner_reset", f"made_by {item.made_by!r} is not a speaker")
            item.made_by = GROUP
        if isinstance(item, OpenQuestion) and item.raised_by not in self.speakers:
            changed = True
            self._issue(kind, label, "owner_reset", f"raised_by {item.raised_by!r} unknown")
            item.raised_by = kept[0].speaker
        if isinstance(item, KeyPoint):
            item.speakers_involved = [s for s in item.speakers_involved if s in self.speakers]
        if isinstance(item, ActionItem) and item.due_date and item.due_date.iso:
            try:
                date.fromisoformat(item.due_date.iso)
            except ValueError:
                item.due_date.iso = None
        if changed and isinstance(item, Decision | ActionItem):
            item.confidence = _DOWNGRADE[item.confidence]
            self.downgraded += 1
        if not changed:
            self.passed += 1
        return item

    def _issue(self, kind: str, label: str, action: str, reason: str) -> None:
        self.issues.append(
            VerificationIssue.model_validate(
                {"item_type": kind, "item": label[:120], "action": action, "reason": reason}
            )
        )

    def verify(
        self, extraction: ChunkExtraction, flags: Sequence[InjectionFlag] = ()
    ) -> tuple[ChunkExtraction, VerificationReport]:
        result = ChunkExtraction(
            key_points=_keep(
                self.check_item("key_point", k.title, k) for k in extraction.key_points
            ),
            decisions=_keep(
                self.check_item("decision", d.decision, d) for d in extraction.decisions
            ),
            action_items=_keep(
                self.check_item("action_item", a.task, a) for a in extraction.action_items
            ),
            open_questions=_keep(
                self.check_item("open_question", q.question, q) for q in extraction.open_questions
            ),
        )
        report = VerificationReport(
            checked=self.checked,
            passed=self.passed,
            dropped=self.dropped,
            downgraded=self.downgraded,
            evidence_checked=self.evidence_checked,
            evidence_passed=self.evidence_passed,
            issues=self.issues,
            injection_flags=list(flags),
        )
        return result, report


def _keep(items: object) -> list:  # type: ignore[type-arg]
    return [i for i in items if i is not None]  # type: ignore[attr-defined]
