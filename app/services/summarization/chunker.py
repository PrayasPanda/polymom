"""Transcript lines for the LLM and token-budgeted chunks on utterance boundaries."""

import math
from collections.abc import Sequence

from app.schemas.transcript import Utterance


def format_timestamp(seconds: float) -> str:
    total = max(0, int(seconds))
    return f"{total // 3600:02d}:{total % 3600 // 60:02d}:{total % 60:02d}"


def sanitize(text: str) -> str:
    """One line, and no angle brackets, so a speaker can't close the <transcript> block."""
    return " ".join(text.replace("<", "‹").replace(">", "›").split())  # noqa: RUF001


def format_line(utterance: Utterance) -> str:
    """``[u42][00:12:05][Person 2][hi] text``: ids and labels the model cites as evidence."""
    return (
        f"[u{utterance.id}][{format_timestamp(utterance.start)}][{utterance.speaker}]"
        f"[{utterance.primary_language or '-'}] {sanitize(utterance.text)}"
    )


def format_transcript(utterances: Sequence[Utterance]) -> str:
    return "\n".join(format_line(u) for u in utterances)


def estimate_tokens(text: str) -> int:
    """Provider-neutral estimate: ~4 chars/token for ASCII, ~2 for Devanagari/Odia.

    ponytail: heuristic, errs high for Indic scripts; swap in the provider tokenizer
    if budgets need to be tight.
    """
    ascii_chars = sum(c.isascii() for c in text)
    return math.ceil(ascii_chars / 4 + (len(text) - ascii_chars) / 2)


def chunk_utterances(
    utterances: Sequence[Utterance], budget: int, overlap: int = 2
) -> list[list[Utterance]]:
    """Consecutive chunks of at most ``budget`` tokens; never splits an utterance.

    Each chunk after the first repeats the last ``overlap`` utterances of the
    previous one, so a decision spanning the boundary is seen whole. A single
    utterance larger than the budget becomes its own chunk.
    """
    chunks: list[list[Utterance]] = []
    current: list[Utterance] = []
    used = 0
    for utterance in utterances:
        cost = estimate_tokens(format_line(utterance)) + 1
        if current and used + cost > budget:
            chunks.append(current)
            current = current[-overlap:] if overlap else []
            used = sum(estimate_tokens(format_line(u)) + 1 for u in current)
            if used + cost > budget:
                current, used = [], 0
        current.append(utterance)
        used += cost
    if current:
        chunks.append(current)
    return chunks
