"""Markdown minutes."""

from app.services.export.document import MinutesDocument, participant_line


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def _table(headers: tuple[str, ...], rows: list[list[str]]) -> list[str]:
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(_cell(c) for c in row) + " |" for row in rows]
    return out


def render_markdown(doc: MinutesDocument) -> bytes:
    out = [
        f"# {doc.title}",
        "",
        f"**Date:** {doc.date}  ",
        f"**Duration:** {doc.duration}  ",
        f"**Languages:** {', '.join(doc.languages) or '-'}  ",
        "**Participants:** " + ("; ".join(participant_line(p) for p in doc.participants) or "-"),
        "",
        "## Executive summary",
        "",
        doc.executive_summary or "_Summary not available._",
        "",
    ]
    if doc.agenda:
        out += ["## Agenda", "", *(f"{i}. {t}" for i, t in enumerate(doc.agenda, 1)), ""]
    if doc.key_points:
        out += ["## Key discussion points", ""]
        out += [f"- **{title}**: {text}" for title, text in doc.key_points]
        out.append("")
    out += ["## Decisions", ""]
    out += _table(doc.DECISION_HEADERS, doc.decisions) if doc.decisions else ["_None recorded._"]
    out += ["", "## Action items", ""]
    out += (
        _table(doc.ACTION_HEADERS, doc.action_items) if doc.action_items else ["_None recorded._"]
    )
    out += ["", "## Open questions", ""]
    out += [f"- {q}" for q in doc.open_questions] or ["_None._"]
    out += ["", "## Appendix: transcript", ""]
    out += [f"**[{t.start}] {t.speaker}:** {t.text}  " for t in doc.transcript] or [
        "_Transcript not available._"
    ]
    if doc.footer:
        out += ["", "---", "", f"_{doc.footer}_"]
    return ("\n".join(out) + "\n").encode("utf-8")
