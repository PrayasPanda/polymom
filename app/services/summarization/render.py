"""Markdown minutes. Speaker display names are applied here, at render time only."""

from app.schemas.summary import Evidence, MeetingSummary
from app.services.summarization.chunker import format_timestamp


def _cite(evidence: list[Evidence], name: dict[str, str]) -> str:
    ev = evidence[0]
    when = format_timestamp(ev.start) if ev.start is not None else "?"
    return f'> "{ev.quote}" ({name.get(ev.speaker, ev.speaker)}, {when})'


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def summary_to_markdown(summary: MeetingSummary, names: dict[str, str] | None = None) -> str:
    n = names or {}

    def who(label: str) -> str:
        return n.get(label, label)

    out = [f"# {summary.title}", ""]
    info = summary.model_info
    out += [
        f"_Generated {summary.generated_at:%Y-%m-%d %H:%M} UTC with {info.provider}/{info.model} "
        f"(prompts {info.prompt_version}). Languages: "
        f"{', '.join(summary.source_languages) or '-'}; summary in {summary.output_language}._",
        "",
        "## Executive summary",
        "",
        summary.executive_summary,
        "",
    ]
    if summary.agenda_topics:
        out += ["## Agenda", "", *(f"{i}. {t}" for i, t in enumerate(summary.agenda_topics, 1)), ""]
    if summary.key_points:
        out += ["## Key discussion points", ""]
        for k in summary.key_points:
            people = ", ".join(who(s) for s in k.speakers_involved)
            out += [f"- **{k.title}**: {k.description}" + (f" ({people})" if people else "")]
            out += ["  " + _cite(k.evidence, n)]
        out.append("")
    if summary.decisions:
        out += [
            "## Decisions",
            "",
            "| # | Decision | By | Confidence | Status |",
            "|---|---|---|---|---|",
        ]
        for i, d in enumerate(summary.decisions, 1):
            status = d.status + (f": {d.note}" if d.note else "")
            out.append(
                f"| {i} | {_cell(d.decision)} | {who(d.made_by)} | {d.confidence} "
                f"| {_cell(status)} |"
            )
        out.append("")
    if summary.action_items:
        out += [
            "## Action items",
            "",
            "| # | Task | Owner | Due | Priority | Confidence |",
            "|---|---|---|---|---|---|",
        ]
        for i, a in enumerate(summary.action_items, 1):
            due = "-"
            if a.due_date:
                due = a.due_date.raw + (f" ({a.due_date.iso})" if a.due_date.iso else "")
            out.append(
                f"| {i} | {_cell(a.task)} | {who(a.owner)} | {_cell(due)} | {a.priority} "
                f"| {a.confidence} |"
            )
        out.append("")
    if summary.open_questions:
        out += ["## Open questions", ""]
        out += [f"- {q.question} ({who(q.raised_by)})" for q in summary.open_questions]
        out.append("")
    r = summary.verification_report
    out += [
        "## Verification",
        "",
        f"{r.checked} items checked: {r.passed} passed, {r.downgraded} downgraded, "
        f"{r.dropped} dropped. Evidence grounding: {r.evidence_passed}/{r.evidence_checked}.",
    ]
    if r.injection_flags:
        out.append(
            f"{len(r.injection_flags)} utterance(s) looked like instructions to the model and "
            "were treated as ordinary speech."
        )
    return "\n".join(out) + "\n"
