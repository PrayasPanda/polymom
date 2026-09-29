"""Async engine and session factory. The backend is chosen by ``DATABASE_URL``."""

from typing import Any

from sqlalchemy import event
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)


def create_engine(url: str) -> AsyncEngine:
    """Create an async engine for any SQLAlchemy URL (SQLite, Postgres via asyncpg, ...).

    SQLite gets ``PRAGMA foreign_keys=ON`` so ``ON DELETE CASCADE`` works as on Postgres.
    """
    engine = create_async_engine(url, pool_pre_ping=True)
    if engine.dialect.name == "sqlite":

        @event.listens_for(engine.sync_engine, "connect")
        def _enable_foreign_keys(dbapi_connection: Any, _record: Any) -> None:
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    return engine


def create_sessionmaker(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False)
