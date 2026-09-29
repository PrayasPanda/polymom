"""Minutes-of-Meeting exports: docx, pdf, md and json."""

from typing import Literal

from app.schemas.result import MeetingResult
from app.services.export.document import build_document

ExportFormat = Literal["docx", "pdf", "md", "json"]

MEDIA_TYPES: dict[str, str] = {
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "pdf": "application/pdf",
    "md": "text/markdown; charset=utf-8",
    "json": "application/json",
}


def render_export(fmt: ExportFormat, result: MeetingResult) -> bytes:
    """Render the full result (all sections included) in ``fmt``."""
    if fmt == "json":
        return result.model_dump_json(indent=2).encode("utf-8")
    doc = build_document(result)
    if fmt == "md":
        from app.services.export.markdown import render_markdown

        return render_markdown(doc)
    if fmt == "docx":
        from app.services.export.docx_export import render_docx

        return render_docx(doc)
    from app.services.export.pdf_export import render_pdf

    return render_pdf(doc)
