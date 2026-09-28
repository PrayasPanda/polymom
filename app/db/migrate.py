"""Run Alembic migrations programmatically (used at startup and in tests)."""

from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import Connection
from sqlalchemy.ext.asyncio import AsyncEngine

MIGRATIONS_DIR = Path(__file__).parent / "migrations"


async def upgrade_to_head(engine: AsyncEngine) -> None:
    """Apply all pending migrations over a connection from ``engine``."""
    cfg = Config()
    cfg.set_main_option("script_location", str(MIGRATIONS_DIR))

    def _upgrade(connection: Connection) -> None:
        cfg.attributes["connection"] = connection
        command.upgrade(cfg, "head")

    async with engine.begin() as conn:
        await conn.run_sync(_upgrade)
