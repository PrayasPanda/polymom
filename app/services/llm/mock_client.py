"""Deterministic, offline LLM for tests and demos.

Two modes:

- **Scripted**: ``MockLLMClient(settings, responses=[...])`` returns the given raw
  strings in order (to exercise the repair loop and error paths).
- **Heuristic** (default): reads the ``[uN][hh:mm:ss][speaker][lang] text`` lines
  of the prompt and builds schema-valid, grounded output with keyword rules
  (agreed/सहमत/ରାଜି -> decision, will/करेंगे/କରିବି -> action item, "?" -> question).
"""

import json
import re
from typing import Any

from pydantic import BaseModel

from app.core.exceptions import LLMError
from app.schemas.summary import ChunkExtraction, SummaryDraft, SummaryHeader
from app.services.llm.base import Completion, LLMClient, Message

LINE = re.compile(r"^\[u(\d+)\]\[[\d:]+\]\[([^\]]+)\]\[[^\]]*\] (.*)$", re.MULTILINE)
DECISION = re.compile(r"\b(agreed|decided|final)\b|सहमत|तय|ରାଜି|ସ୍ଥିର", re.IGNORECASE)
ACTION = re.compile(
    r"\b(will|i'll|going to)\b|करेंगे|करूंगा|करूँगा|भेजूंगा|भेज दूंगा|କରିବି|ପଠାଇବି|କରିବେ",
    re.IGNORECASE,
)
DUE = re.compile(
    r"\b(tomorrow|today|kal|parso|agle hafte|next week|by eod|eod|monday|tuesday|wednesday|"
    r"thursday|friday)\b|कल|परसों|अगले हफ्ते|शुक्रवार|ଆସନ୍ତାକାଲି|ଶୁକ୍ରବାର",
    re.IGNORECASE,
)
TOPIC = re.compile(r"\b(item|agenda|topic)\b|मुद्दा|ବିଷୟ", re.IGNORECASE)


def _lines(prompt: str) -> list[tuple[int, str, str]]:
    return [(int(i), spk, text.strip()) for i, spk, text in LINE.findall(prompt)]


def _evidence(uid: int, speaker: str, text: str) -> dict[str, Any]:
    return {"utterance_ids": [uid], "speaker": speaker, "quote": text[:80]}


def _short(text: str, words: int = 8) -> str:
    return " ".join(text.split()[:words]).rstrip(".,।?!")


def heuristic_extraction(prompt: str) -> dict[str, Any]:
    out: dict[str, Any] = {"key_points": [], "decisions": [], "action_items": []}
    out["open_questions"] = []
    seen_speakers: set[str] = set()
    for uid, speaker, text in _lines(prompt):
        ev = [_evidence(uid, speaker, text)]
        if speaker not in seen_speakers:
            seen_speakers.add(speaker)
            out["key_points"].append(
                {
                    "title": _short(text, 6),
                    "description": text,
                    "speakers_involved": [speaker],
                    "evidence": ev,
                }
            )
        if DECISION.search(text):
            out["decisions"].append(
                {
                    "decision": text,
                    "topic": _short(text, 4),
                    "made_by": "group",
                    "rationale": "",
                    "evidence": ev,
                    "confidence": "high",
                }
            )
        elif ACTION.search(text) and not text.endswith("?"):
            due = DUE.search(text)
            out["action_items"].append(
                {
                    "task": text,
                    "owner": speaker,
                    "due_date": {"raw": due.group(0), "iso": None} if due else None,
                    "priority": "medium",
                    "evidence": ev,
                    "confidence": "medium",
                }
            )
        if text.endswith("?"):
            out["open_questions"].append({"question": text, "raised_by": speaker, "evidence": ev})
    return out


def heuristic_header(prompt: str) -> dict[str, Any]:
    lines = _lines(prompt)
    topics = [_short(t, 6) for _, _, t in lines if TOPIC.search(t)][:5]
    speakers = sorted({s for _, s, _ in lines})
    return {
        "title": topics[0] if topics else "Meeting summary",
        "executive_summary": " ".join(
            [
                "This is a deterministic mock summary.",
                f"The transcript excerpt has {len(lines)} utterances.",
                f"Participants: {', '.join(speakers) or 'none'}.",
                f"Agenda topics detected: {len(topics)}.",
                "Decisions and action items are listed below with their evidence.",
            ]
        ),
        "agenda_topics": topics,
        "overall_sentiment": "neutral",
    }


class MockLLMClient(LLMClient):
    provider = "mock"

    def __init__(self, *args: Any, responses: list[str] | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.responses = list(responses) if responses is not None else None
        self.requests: list[tuple[str, list[Message]]] = []

    async def _complete(
        self, system: str, messages: list[Message], schema: type[BaseModel], temperature: float
    ) -> Completion:
        self.requests.append((system, list(messages)))
        prompt = messages[0].content
        if self.responses is not None:
            if not self.responses:
                raise LLMError("MockLLMClient ran out of scripted responses.")
            text = self.responses.pop(0)
        elif issubclass(schema, SummaryDraft):
            text = json.dumps(heuristic_header(prompt) | heuristic_extraction(prompt))
        elif issubclass(schema, ChunkExtraction):
            text = json.dumps(heuristic_extraction(prompt))
        elif issubclass(schema, SummaryHeader):
            text = json.dumps(heuristic_header(prompt))
        else:
            raise LLMError(f"MockLLMClient has no heuristic for {schema.__name__}.")
        return Completion(text, len(prompt) // 4, len(text) // 4)
