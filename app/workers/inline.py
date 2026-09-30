"""``PIPELINE_EXECUTION=inline``: run the pipeline in the API process (tests, local debug).

Never used in production: the fail-fast settings check requires the queue there.
"""

import uuid

from app.core.config import Settings
from app.pipelines.mom_pipeline import MoMPipeline
from app.workers.webhook import deliver_webhook


async def run_inline(
    pipeline: MoMPipeline,
    settings: Settings,
    meeting_id: uuid.UUID,
    *,
    from_stage: str | None = None,
    callback_url: str | None = None,
) -> None:
    status = await pipeline.run(meeting_id, from_stage=from_stage)
    if status is not None and callback_url:
        await deliver_webhook(settings, meeting_id, status.value, callback_url)
