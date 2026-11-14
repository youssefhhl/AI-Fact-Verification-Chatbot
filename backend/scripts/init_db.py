"""Create or upgrade all database tables (explicit step, not run at app startup).

1. Alembic migrations: application tables (conversations, messages).
2. LangGraph's AsyncPostgresSaver.setup(): checkpoint tables (idempotent).

Usage (from backend/, with PostgreSQL running):
    uv run python -m scripts.init_db
"""

import asyncio
import logging
from pathlib import Path

from alembic import command
from alembic.config import Config
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from app.config import get_settings
from app.database.session import checkpointer_conninfo

ALEMBIC_INI = Path(__file__).resolve().parent.parent / "alembic.ini"


def run_migrations(database_url: str) -> None:
    config = Config(str(ALEMBIC_INI))
    config.attributes["database_url"] = database_url
    config.attributes["configure_logging"] = False
    command.upgrade(config, "head")


async def setup_checkpointer(database_url: str) -> None:
    async with AsyncPostgresSaver.from_conn_string(checkpointer_conninfo(database_url)) as saver:
        await saver.setup()


def init_database(database_url: str) -> None:
    run_migrations(database_url)
    asyncio.run(setup_checkpointer(database_url))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    url = get_settings().database_url
    if url is None:
        raise SystemExit("DATABASE_URL is not set (see .env.example)")
    init_database(url.get_secret_value())
    logging.info("Database ready: application tables migrated, LangGraph checkpoint tables set up.")


if __name__ == "__main__":
    main()
