"""PDF minutes via fpdf2 with HarfBuzz text shaping and bundled Noto fonts.

Why fpdf2 rather than WeasyPrint or ReportLab: Hindi and Odia need OpenType shaping
(conjuncts, pre-base vowel signs). fpdf2 shapes through ``uharfbuzz``, a pure-wheel
dependency; WeasyPrint needs the Pango/GObject system libraries (heavy in Docker,
awkward on Windows), and ReportLab's open-source edition has no dependable Indic
shaping. Fonts (SIL OFL 1.1, see ``fonts/OFL.txt``) are embedded and subset, so PDFs
render identically everywhere. Latin text uses Noto Sans; Devanagari and Odia fall
back to Noto Sans Devanagari / Noto Sans Oriya automatically.
"""

from pathlib import Path

from fpdf import FPDF
from fpdf.enums import XPos, YPos

from app.services.export.document import MinutesDocument, participant_line

FONTS_DIR = Path(__file__).parent / "fonts"
FONT_FILES = {
    "Noto": "NotoSans",
    "NotoDeva": "NotoSansDevanagari",
    "NotoOrya": "NotoSansOriya",
}


class MinutesPDF(FPDF):
    def __init__(self, title: str) -> None:
        super().__init__(format="A4")
        self.doc_title = title
        for family, stem in FONT_FILES.items():
            self.add_font(family, "", FONTS_DIR / f"{stem}-Regular.ttf")
            self.add_font(family, "B", FONTS_DIR / f"{stem}-Bold.ttf")
        self.set_font("Noto", size=10)
        self.set_fallback_fonts(["NotoDeva", "NotoOrya"], exact_match=False)
        self.set_text_shaping(True)
        self.set_auto_page_break(auto=True, margin=15)
        self.set_title(title)
        self.set_creator("polymom")

    def footer(self) -> None:
        self.set_y(-12)
        self.set_font("Noto", size=8)
        self.set_text_color(120)
        self.cell(0, 8, f"{self.doc_title}  ·  page {self.page_no()}", align="C")
        self.set_text_color(0)

    def heading(self, text: str, size: int = 13) -> None:
        self.ln(3)
        self.set_font("Noto", "B", size)
        self.multi_cell(0, size * 0.55, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_font("Noto", size=10)
        self.ln(1)

    def para(self, text: str, height: float = 5.5) -> None:
        self.multi_cell(0, height, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def label_line(self, label: str, value: str) -> None:
        self.set_font("Noto", "B", 10)
        self.write(5.5, f"{label}: ")
        self.set_font("Noto", size=10)
        self.write(5.5, value)
        self.ln(6)

    def grid(
        self, headers: tuple[str, ...], rows: list[list[str]], widths: tuple[int, ...]
    ) -> None:
        self.set_font("Noto", size=9)
        with self.table(col_widths=widths, text_align="LEFT", line_height=5) as table:
            table.row(headers)
            for row in rows:
                table.row(row)
        self.set_font("Noto", size=10)


def render_pdf(doc: MinutesDocument) -> bytes:
    pdf = MinutesPDF(doc.title)
    pdf.add_page()
    pdf.heading(doc.title, size=18)
    pdf.label_line("Date", doc.date)
    pdf.label_line("Duration", doc.duration)
    pdf.label_line("Languages", ", ".join(doc.languages) or "-")
    pdf.label_line("Participants", "; ".join(participant_line(p) for p in doc.participants) or "-")

    pdf.heading("Executive summary")
    pdf.para(doc.executive_summary or "Summary not available.")
    if doc.agenda:
        pdf.heading("Agenda")
        for i, topic in enumerate(doc.agenda, 1):
            pdf.para(f"{i}. {topic}")
    if doc.key_points:
        pdf.heading("Key discussion points")
        for title, text in doc.key_points:
            pdf.para(f"• {title}: {text}")
    pdf.heading("Decisions")
    if doc.decisions:
        pdf.grid(doc.DECISION_HEADERS, doc.decisions, (6, 60, 18, 16, 30))
    else:
        pdf.para("None recorded.")
    pdf.heading("Action items")
    if doc.action_items:
        pdf.grid(doc.ACTION_HEADERS, doc.action_items, (6, 62, 20, 26, 16))
    else:
        pdf.para("None recorded.")
    pdf.heading("Open questions")
    for question in doc.open_questions or ["None."]:
        pdf.para(f"• {question}")

    pdf.add_page()
    pdf.heading("Appendix: transcript")
    for line in doc.transcript:
        pdf.set_font("Noto", "B", 9)
        pdf.write(5, f"[{line.start}] {line.speaker}: ")
        pdf.set_font("Noto", size=9)
        pdf.write(5, line.text)
        pdf.ln(6)
    if not doc.transcript:
        pdf.para("Transcript not available.")
    if doc.footer:
        pdf.ln(4)
        pdf.set_font("Noto", size=8)
        pdf.para(doc.footer, 4.5)
    return bytes(pdf.output())
