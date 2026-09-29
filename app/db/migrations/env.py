"""Alembic environment supporting both the CLI and programmatic (in-app) runs."""

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy import Connection

import app.models  # noqa: F401 - register ORM models on Base.metadata
from app.core.config import get_settings
from app.db.base import Base
from app.db.session import create_engine

config = context.config
target_metadata = Base.metadata


def _do_run_migrations(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        render_as_batch=connection.dialect.name == "sqlite",
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_offline() -> None:
    context.configure(
        url=config.get_main_option("sqlalchemy.url") or get_settings().resolved_database_url,
        target_metadata=target_metadata,
        literal_binds=True,
    )
    with context.begin_transaction():
        context.run_migrations()


async def _run_async_migrations() -> None:
    url = config.get_main_option("sqlalchemy.url") or get_settings().resolved_database_url
    engine = create_engine(url)
    async with engine.connect() as connection:
        if connection.dialect.name == "sqlite":
            await connection.exec_driver_sql("PRAGMA foreign_keys=OFF")  # see app.db.migrate
        await connection.run_sync(_do_run_migrations)
        await connection.commit()
    await engine.dispose()


def run_migrations_online() -> None:
    connection = config.attributes.get("connection")
    if connection is not None:
        # Invoked from app.db.migrate with an existing connection.
        _do_run_migrations(connection)
        return
    if config.config_file_name is not None:
        fileConfig(config.config_file_name)
    asyncio.run(_run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
