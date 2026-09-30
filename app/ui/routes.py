"""Web UI: Jinja2 pages plus HTMX partials, served by the API process at ``/``.

Page shells carry no data and are public. Every partial and every action goes
through the same ``X-API-Key`` auth as the JSON API: ``app.js`` adds the header
to HTMX requests and to its own ``fetch`` calls.
"""

from pathlib import Path
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.api.auth import CurrentApiKey, resolve_api_key
from app.api.deps import MeetingServiceDep, ResultServiceDep, SettingsDep
from app.api.v1.routes.status import get_status
from app.repositories.meeting_repository import MeetingQuery
from app.schemas.meeting import MeetingStatus

TEMPLATES_DIR = Path(__file__).parent / "templates"
STATIC_DIR = Path(__file__).parent / "static"
FONTS_DIR = Path(__file__).parents[1] / "services" / "export" / "fonts"

# Pages are HTML, so they get their own CSP instead of the API's default-src 'none'.
# No inline scripts or style attributes: charts are SVG, colours are CSS classes.
UI_CSP = (
    "default-src 'self'; img-src 'self' data: blob:; media-src 'self' blob:; "
    "font-src 'self'; style-src 'self'; script-src 'self'; connect-src 'self'; "
    "frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
)
SPEAKER_COLOURS = 8
ACTIVE = {MeetingStatus.QUEUED, MeetingStatus.PROCESSING}

templates = Jinja2Templates(directory=TEMPLATES_DIR)
router = APIRouter(include_in_schema=False)
partials = APIRouter(
    prefix="/ui/partials", dependencies=[Depends(resolve_api_key)], include_in_schema=False
)


def _render(request: Request, name: str, **context: Any) -> HTMLResponse:
    response = templates.TemplateResponse(request, name, context)
    response.headers["Content-Security-Policy"] = UI_CSP
    return response


def fmt_time(seconds: float | None) -> str:
    """``75.4`` -> ``"01:15"``; hours are shown only when needed."""
    if seconds is None:
        return "--:--"
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


templates.env.filters["mmss"] = fmt_time


@router.get("/", response_class=HTMLResponse)
async def index(request: Request, settings: SettingsDep) -> HTMLResponse:
    return _render(request, "index.html", settings=settings)


@router.get("/meetings/{meeting_id}", response_class=HTMLResponse)
async def meeting_page(request: Request, meeting_id: UUID, settings: SettingsDep) -> HTMLResponse:
    return _render(request, "meeting.html", settings=settings, meeting_id=meeting_id)


@partials.get("/meetings", response_class=HTMLResponse)
async def meetings_partial(request: Request, service: MeetingServiceDep) -> HTMLResponse:
    page = await service.page(MeetingQuery(limit=50))
    items = [await service.to_read(m) for m in page.items]
    polling = any(m.status in ACTIVE for m in items)
    return _render(
        request, "partials/meetings.html", items=items, total=page.total, polling=polling
    )


@partials.get("/meetings/{meeting_id}", response_class=HTMLResponse)
async def meeting_partial(
    request: Request,
    meeting_id: UUID,
    service: MeetingServiceDep,
    results: ResultServiceDep,
    settings: SettingsDep,
    key: CurrentApiKey,
) -> HTMLResponse:
    """Progress while the pipeline runs (HTMX re-polls), the full minutes once it is done."""
    meeting = await service.get(meeting_id)
    if meeting.status in ACTIVE:
        progress = await get_status(meeting_id, service, settings, key)
        return _render(request, "partials/progress.html", meeting=meeting, progress=progress)
    result = await results.result(meeting_id)
    labels = [s.label for s in result.speakers]
    rows = {label: i for i, label in enumerate(labels)}
    colours = {label: i % SPEAKER_COLOURS for label, i in rows.items()}
    names = {s.label: s.display_name or s.label for s in result.speakers}
    return _render(
        request,
        "partials/detail.html",
        result=result,
        colours=colours,
        rows=rows,
        names=names,
        duration=(result.meeting.duration_seconds or 0.0) or 1.0,
    )
