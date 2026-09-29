"""Run Alembic migrations programmatically (used at startup and in tests)."""

from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import Connection
from sqlalchemy.ext.asyncio import AsyncEngine

MIGRATIONS_DIR = Path(__file__).parent / "migrations"


async def _run(engine: AsyncEngine, action: str, revision: str) -> None:
    cfg = Config()
    cfg.set_main_option("script_location", str(MIGRATIONS_DIR))

    def _migrate(connection: Connection) -> None:
        cfg.attributes["connection"] = connection
        getattr(command, action)(cfg, revision)

    async with engine.connect() as conn:
        sqlite = conn.dialect.name == "sqlite"
        if sqlite:
            # Batch migrations rebuild tables on SQLite; with foreign keys on, dropping
            # the old table would cascade-delete child rows. Must run outside a transaction.
            await conn.exec_driver_sql("PRAGMA foreign_keys=OFF")
            await conn.commit()
        try:
            async with conn.begin():
                await conn.run_sync(_migrate)
        finally:
            if sqlite:
                await conn.exec_driver_sql("PRAGMA foreign_keys=ON")
                await conn.commit()


async def upgrade_to_head(engine: AsyncEngine) -> None:
    """Apply all pending migrations over a connection from ``engine``."""
    await _run(engine, "upgrade", "head")


async def upgrade_to(engine: AsyncEngine, revision: str) -> None:
    await _run(engine, "upgrade", revision)


async def downgrade_to(engine: AsyncEngine, revision: str) -> None:
    await _run(engine, "downgrade", revision)
