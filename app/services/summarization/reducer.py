"""Reduce step: merge map outputs, de-duplicate and resolve superseded decisions.

Deterministic, so it is cheap and testable; the LLM only writes the narrative
(title, executive summary, agenda) from the merged items.

- Duplicates: text similarity (rapidfuzz ``token_set_ratio``) >= ``DUPLICATE_SCORE``,
  or >= ``RELATED_SCORE`` when the items cite the same utterances (chunk overlap).
  Merged items keep the union of their evidence.
- Supersession: two different decisions on the same ``topic`` -> the later one
  (by first cited utterance) stays active, the earlier becomes ``superseded``,
  and both get a ``note``.
"""

from collections.abc import Callable, Sequence
from typing import TypeVar

from rapidfuzz import fuzz

from app.schemas.summary import (
    UNASSIGNED,
    ActionItem,
    ChunkExtraction,
    Decision,
    Evidence,
    KeyPoint,
    OpenQuestion,
)
from app.services.summarization.verifier import normalize

DUPLICATE_SCORE = 85.0
RELATED_SCORE = 70.0
TOPIC_SCORE = 85.0
_RANK = {"low": 0, "medium": 1, "high": 2}

Item = TypeVar("Item", KeyPoint, Decision, ActionItem, OpenQuestion)


def similarity(a: str, b: str) -> float:
    return float(fuzz.token_set_ratio(normalize(a), normalize(b)))


def first_utterance(item: KeyPoint | Decision | ActionItem | OpenQuestion) -> int:
    return min((i for ev in item.evidence for i in ev.utterance_ids), default=0)


def merge_evidence(a: Sequence[Evidence], b: Sequence[Evidence]) -> list[Evidence]:
    seen = {tuple(ev.utterance_ids) for ev in a}
    return [*a, *(ev for ev in b if tuple(ev.utterance_ids) not in seen)]


def _is_duplicate(a: str, b: str, ev_a: Sequence[Evidence], ev_b: Sequence[Evidence]) -> bool:
    score = similarity(a, b)
    if score >= DUPLICATE_SCORE:
        return True
    ids_a = {i for ev in ev_a for i in ev.utterance_ids}
    shared = any(i in ids_a for ev in ev_b for i in ev.utterance_ids)
    return shared and score >= RELATED_SCORE


def _dedupe(
    items: Sequence[Item],
    key: Callable[[Item], str],
    compatible: Callable[[Item, Item], bool],
    merge: Callable[[Item, Item], None],
) -> list[Item]:
    kept: list[Item] = []
    for item in sorted(items, key=first_utterance):
        match = next(
            (
                k
                for k in kept
                if compatible(k, item)
                and _is_duplicate(key(k), key(item), k.evidence, item.evidence)
            ),
            None,
        )
        if match is None:
            kept.append(item.model_copy(deep=True))
        else:
            match.evidence = merge_evidence(match.evidence, item.evidence)
            merge(match, item)
    return kept


def _merge_key_point(kept: KeyPoint, new: KeyPoint) -> None:
    kept.speakers_involved = list(dict.fromkeys([*kept.speakers_involved, *new.speakers_involved]))


def _merge_decision(kept: Decision, new: Decision) -> None:
    if _RANK[new.confidence] > _RANK[kept.confidence]:
        kept.confidence = new.confidence
    kept.rationale = kept.rationale or new.rationale


def _merge_action(kept: ActionItem, new: ActionItem) -> None:
    if kept.owner == UNASSIGNED:
        kept.owner = new.owner
    kept.due_date = kept.due_date or new.due_date
    if _RANK[new.priority] > _RANK[kept.priority]:
        kept.priority = new.priority


def _same_owner(a: ActionItem, b: ActionItem) -> bool:
    return a.owner == b.owner or UNASSIGNED in (a.owner, b.owner)


def resolve_supersession(decisions: list[Decision]) -> list[Decision]:
    """Later decisions on the same topic supersede earlier ones (list sorted by time)."""
    for i, later in enumerate(decisions):
        for earlier in decisions[:i]:
            if (
                earlier.status == "active"
                and earlier.topic
                and later.topic
                and similarity(earlier.topic, later.topic) >= TOPIC_SCORE
            ):
                earlier.status = "superseded"
                earlier.note = f"Superseded later in the meeting by: {later.decision}"
                later.note = f"Replaces an earlier decision: {earlier.decision}"
    return decisions


def merge_extractions(parts: Sequence[ChunkExtraction]) -> ChunkExtraction:
    def always(a: object, b: object) -> bool:
        return True

    return ChunkExtraction(
        key_points=_dedupe(
            [k for p in parts for k in p.key_points], lambda k: k.title, always, _merge_key_point
        ),
        decisions=resolve_supersession(
            _dedupe(
                [d for p in parts for d in p.decisions],
                lambda d: d.decision,
                always,
                _merge_decision,
            )
        ),
        action_items=_dedupe(
            [a for p in parts for a in p.action_items], lambda a: a.task, _same_owner, _merge_action
        ),
        open_questions=_dedupe(
            [q for p in parts for q in p.open_questions],
            lambda q: q.question,
            always,
            lambda kept, new: None,
        ),
    )
