"""Background task entry points.

TODO: move to a real job queue (e.g. arq/RQ/Celery) with retries.
"""


async def process_meeting(meeting_id: str) -> None:
    """Run the MoM pipeline for the given meeting outside the request cycle."""
    raise NotImplementedError
