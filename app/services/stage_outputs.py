"""Read a stage's stored output, inline or from the artifact store."""

import json
import uuid
from typing import Any

from app.repositories.artifacts import ArtifactStore
from app.repositories.results_repository import ResultsRepository


async def load_stage_output(
    results: ResultsRepository, store: ArtifactStore, run_id: uuid.UUID, stage: str
) -> Any | None:
    """The JSON output of a completed stage, or ``None`` if it didn't run or failed."""
    row = await results.stage_result(run_id, stage)
    if row is None or row.status != "completed":
        return None
    if row.output_ref:
        return json.loads(await store.get(row.output_ref))
    return row.output
