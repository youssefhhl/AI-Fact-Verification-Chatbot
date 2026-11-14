"""Alembic environment: application tables only (conversations, messages).

LangGraph checkpoint tables are created by AsyncPostgresSaver.setup() instead
(see scripts/init_db.py), so they are excluded from autogenerate here.
"""

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import create_async_engine

from app.config import get_settings
from app.database.models import Base

config = context.config
if config.config_file_name is not None and config.attributes.get("configure_logging", True):
    fileConfig(config.config_file_name, disable_existing_loggers=False)

target_metadata = Base.metadata
CHECKPOINT_TABLES = {"checkpoints", "checkpoint_blobs", "checkpoint_writes", "checkpoint_migrations"}


def database_url() -> str:
    # An explicit URL (e.g. from tests) wins over the application setting.
    url = config.attributes.get("database_url") or config.get_main_option("sqlalchemy.url")
    if url:
        return url
    settings_url = get_settings().database_url
    if settings_url is None:
        raise RuntimeError("DATABASE_URL is not set")
    return settings_url.get_secret_value()


def include_object(obj, name, type_, reflected, compare_to) -> bool:  # noqa: ANN001
    return not (type_ == "table" and name in CHECKPOINT_TABLES)


def run_migrations_offline() -> None:
    context.configure(url=database_url(), target_metadata=target_metadata, literal_binds=True,
                      include_object=include_object, dialect_opts={"paramstyle": "named"})
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata, include_object=include_object)
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    engine = create_async_engine(database_url(), poolclass=pool.NullPool)
    async with engine.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_async_migrations())
