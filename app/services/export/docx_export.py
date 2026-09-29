"""Word (.docx) minutes via python-docx.

Runs set both the Latin and the complex-script font (``w:cs``) to Noto families, so
Word picks a font with Devanagari/Odia glyphs even where the default theme lacks them.
"""

import io

from docx import Document
from docx.document import Document as DocxDocument
from docx.oxml.ns import qn
from docx.shared import Pt

from app.services.export.document import MinutesDocument, participant_line

LATIN_FONT = "Noto Sans"
COMPLEX_FONT = "Noto Sans Devanagari"


def _set_fonts(document: DocxDocument) -> None:
    for style_name in ("Normal", "Title", "Heading 1", "Heading 2", "Table Grid"):
        try:
            style = document.styles[style_name]
        except KeyError:  # pragma: no cover - depends on the template
            continue
        style.font.name = LATIN_FONT
        rpr = style.element.get_or_add_rPr()
        fonts = rpr.find(qn("w:rFonts"))
        if fonts is None:
            fonts = rpr.makeelement(qn("w:rFonts"), {})
            rpr.append(fonts)
        fonts.set(qn("w:ascii"), LATIN_FONT)
        fonts.set(qn("w:hAnsi"), LATIN_FONT)
        fonts.set(qn("w:cs"), COMPLEX_FONT)
    document.styles["Normal"].font.size = Pt(10.5)


def _table(document: DocxDocument, headers: tuple[str, ...], rows: list[list[str]]) -> None:
    table = document.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    for cell, header in zip(table.rows[0].cells, headers, strict=True):
        cell.text = ""
        cell.paragraphs[0].add_run(header).bold = True
    for row in rows:
        for cell, value in zip(table.add_row().cells, row, strict=True):
            cell.text = value


def render_docx(doc: MinutesDocument) -> bytes:
    document = Document()
    _set_fonts(document)
    document.add_heading(doc.title, level=0)
    meta = document.add_paragraph()
    for label, value in (
        ("Date", doc.date),
        ("Duration", doc.duration),
        ("Languages", ", ".join(doc.languages) or "-"),
    ):
        meta.add_run(f"{label}: ").bold = True
        meta.add_run(f"{value}\n")
    meta.add_run("Participants: ").bold = True
    meta.add_run("; ".join(participant_line(p) for p in doc.participants) or "-")

    document.add_heading("Executive summary", level=1)
    document.add_paragraph(doc.executive_summary or "Summary not available.")
    if doc.agenda:
        document.add_heading("Agenda", level=1)
        for topic in doc.agenda:
            document.add_paragraph(topic, style="List Number")
    if doc.key_points:
        document.add_heading("Key discussion points", level=1)
        for title, text in doc.key_points:
            p = document.add_paragraph(style="List Bullet")
            p.add_run(f"{title}: ").bold = True
            p.add_run(text)
    document.add_heading("Decisions", level=1)
    if doc.decisions:
        _table(document, doc.DECISION_HEADERS, doc.decisions)
    else:
        document.add_paragraph("None recorded.")
    document.add_heading("Action items", level=1)
    if doc.action_items:
        _table(document, doc.ACTION_HEADERS, doc.action_items)
    else:
        document.add_paragraph("None recorded.")
    document.add_heading("Open questions", level=1)
    for question in doc.open_questions or ["None."]:
        document.add_paragraph(question, style="List Bullet")
    document.add_page_break()  # type: ignore[no-untyped-call]
    document.add_heading("Appendix: transcript", level=1)
    for line in doc.transcript:
        p = document.add_paragraph()
        p.add_run(f"[{line.start}] {line.speaker}: ").bold = True
        p.add_run(line.text)
    if not doc.transcript:
        document.add_paragraph("Transcript not available.")
    if doc.footer:
        document.add_paragraph().add_run(doc.footer).italic = True
    buf = io.BytesIO()
    document.save(buf)
    return buf.getvalue()
