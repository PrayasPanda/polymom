"""Cross-meeting search and published schemas."""

from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Query

from app.api.deps import ResultServiceDep
from app.schemas.result import SCHEMA_VERSION, MeetingResult, SearchResults

router = APIRouter(tags=["search"])


@router.get("/search", response_model=SearchResults, summary="Full-text search across meetings")
async def search(
    results: ResultServiceDep,
    q: Annotated[str, Query(min_length=1, max_length=200, description="Words in any script.")],
    meeting_id: Annotated[UUID | None, Query(description="Limit to one meeting.")] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
) -> SearchResults:
    """Utterances and summaries of each meeting's latest results, in English, Hindi and Odia."""
    return await results.search(q.strip(), limit, meeting_id)


@router.get(
    "/schema/meeting-result",
    summary="JSON Schema of the consolidated MeetingResult",
    response_model=dict[str, Any],
)
async def meeting_result_schema() -> dict[str, Any]:
    """For integrators: validate ``GET /meetings/{id}/result`` responses or generate clients."""
    schema = MeetingResult.model_json_schema(mode="serialization")
    schema["$id"] = f"https://polymom.dev/schemas/meeting-result/{SCHEMA_VERSION}.json"
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["version"] = SCHEMA_VERSION
    return schema
