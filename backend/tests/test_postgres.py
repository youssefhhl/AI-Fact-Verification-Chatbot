"""Integration tests against a real PostgreSQL (TEST_DATABASE_URL, e.g. the
truthlens_test database from docker-compose). Skipped when it is not reachable.

They cover what SQLite + the in-memory saver cannot: Alembic migrations, the
LangGraph PostgreSQL checkpointer, and state surviving an application restart.
"""

import uuid

import psycopg
import pytest

from app.config import get_settings
from app.database.session import checkpointer_conninfo
from app.services.conversation_service import open_conversation_service
from scripts.init_db import init_database
from tests.fakes import ELDERLY, FakeStack

pytestmark = pytest.mark.anyio


def _test_url() -> str | None:
    url = get_settings().test_database_url
    if url is None:
        return None
    try:
        with psycopg.connect(checkpointer_conninfo(url.get_secret_value()), connect_timeout=2):
            return url.get_secret_value()
    except psycopg.Error:
        return None


TEST_URL = _test_url()
requires_postgres = pytest.mark.skipif(TEST_URL is None, reason="TEST_DATABASE_URL not set or unreachable")


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def database() -> str:
    """Fresh schema, then the same explicit init as production (migrations + checkpointer setup)."""
    assert TEST_URL is not None
    with psycopg.connect(checkpointer_conninfo(TEST_URL), autocommit=True) as conn:
        conn.execute("DROP SCHEMA public CASCADE")
        conn.execute("CREATE SCHEMA public")
    init_database(TEST_URL)
    return TEST_URL


def tables(url: str) -> set[str]:
    with psycopg.connect(checkpointer_conninfo(url)) as conn:
        rows = conn.execute("SELECT tablename FROM pg_tables WHERE schemaname = 'public'").fetchall()
    return {r[0] for r in rows}


def checkpoint_rows(url: str, thread_id: str) -> int:
    with psycopg.connect(checkpointer_conninfo(url)) as conn:
        return conn.execute("SELECT count(*) FROM checkpoints WHERE thread_id = %s", (thread_id,)).fetchone()[0]


@requires_postgres
def test_init_creates_app_and_checkpoint_tables_and_is_idempotent(database) -> None:
    expected = {"alembic_version", "conversations", "messages",
                "checkpoints", "checkpoint_blobs", "checkpoint_writes", "checkpoint_migrations"}
    assert expected <= tables(database)
    init_database(database)  # rerun is a no-op
    assert expected <= tables(database)


@requires_postgres
async def test_context_survives_restart_and_conversations_stay_isolated(database) -> None:
    stack = FakeStack()

    # "Process 1": start a conversation, then shut everything down.
    async with open_conversation_service(database, stack.factory) as service:
        cid, first = await service.chat("Vitamin D prevents bone fractures")
        other_id, _ = await service.chat("Kubernetes is written in Go")
    assert first.decision == "NEW_VERIFICATION"
    assert checkpoint_rows(database, str(cid)) > 0

    # "Process 2": new engine, new pool, new graph. Memory comes only from PostgreSQL.
    stack2 = FakeStack()
    async with open_conversation_service(database, stack2.factory) as service:
        _, follow = await service.chat("What about elderly people?", cid)
        conversation, messages = await service.get_conversation(cid)
        _, other_messages = await service.get_conversation(other_id)

    assert follow.decision == "TARGETED_RESEARCH" and follow.claim == ELDERLY
    assert [t.message for t in follow.state.turns] == ["Vitamin D prevents bone fractures",
                                                       "What about elderly people?"]
    assert "Kubernetes" not in follow.state.model_dump_json()  # no leak from the other thread
    assert [m.role for m in messages] == ["user", "assistant", "user", "assistant"]
    assert messages[3].verification["claim"] == ELDERLY
    assert len(other_messages) == 2


@requires_postgres
async def test_delete_removes_rows_and_checkpoints(database) -> None:
    stack = FakeStack()
    async with open_conversation_service(database, stack.factory) as service:
        cid, _ = await service.chat("Vitamin D prevents bone fractures")
        keep, _ = await service.chat("Kubernetes is written in Go")
        await service.delete_conversation(cid)
        assert [c.id for c, _ in await service.list_conversations()] == [keep]
    assert checkpoint_rows(database, str(cid)) == 0
    assert checkpoint_rows(database, str(keep)) > 0
    with psycopg.connect(checkpointer_conninfo(database)) as conn:
        assert conn.execute("SELECT count(*) FROM messages WHERE conversation_id = %s",
                            (uuid.UUID(str(cid)),)).fetchone()[0] == 0
