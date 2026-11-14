"""Chat use cases: persist messages, run the LangGraph workflow, manage conversations.

Two stores, one id:
- PostgreSQL tables `conversations` / `messages`: history shown to the user.
- LangGraph checkpoints (thread_id = conversation id): the agent's working memory
  (topics, evidence, verdicts) restored on every turn.
"""

import asyncio
import logging
import uuid
import weakref
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

import psycopg
from langgraph.checkpoint.base import BaseCheckpointSaver
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.agents.fact_checking_agent import FactCheckingAgent
from app.database import repository
from app.database.models import Conversation, Message
from app.graph.workflow import build_graph, output_from_state
from app.schemas.conversation import FactCheckOutput
from app.services.llm_service import LLMConfigurationError

logger = logging.getLogger(__name__)
TITLE_CHARS = 80
DATABASE_ERRORS = (SQLAlchemyError, psycopg.Error)


class ConversationError(Exception):
    """Base class for errors the API turns into HTTP responses."""


class ConversationNotFoundError(ConversationError):
    pass


class AgentUnavailableError(ConversationError):
    """The agent cannot be built (e.g. GROQ_API_KEY missing)."""


class AgentFailedError(ConversationError):
    """The workflow crashed while answering. The user message is already stored."""

    def __init__(self, message: str, conversation_id: uuid.UUID) -> None:
        super().__init__(message)
        self.conversation_id = conversation_id


class DatabaseUnavailableError(ConversationError):
    pass


class ConversationService:
    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        checkpointer: BaseCheckpointSaver,
        agent_factory: Callable[[], FactCheckingAgent] = FactCheckingAgent.from_settings,
    ) -> None:
        self.session_factory = session_factory
        self.checkpointer = checkpointer
        self.agent_factory = agent_factory
        self._graph: Any = None
        # One turn at a time per conversation (in this process), so two requests for the
        # same thread cannot interleave their checkpoints and messages.
        self._locks: weakref.WeakValueDictionary[uuid.UUID, asyncio.Lock] = weakref.WeakValueDictionary()

    def _lock(self, conversation_id: uuid.UUID) -> asyncio.Lock:
        lock = self._locks.get(conversation_id)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[conversation_id] = lock
        return lock

    def graph(self) -> Any:
        if self._graph is None:  # built lazily: the app must start without a Groq key
            try:
                agent = self.agent_factory()
            except LLMConfigurationError as exc:
                raise AgentUnavailableError(str(exc)) from exc
            self._graph = build_graph(agent, self.checkpointer)
        return self._graph

    # --- Chat ------------------------------------------------------------------------

    async def chat(self, message: str, conversation_id: uuid.UUID | None = None) -> tuple[uuid.UUID, FactCheckOutput]:
        graph = self.graph()  # fail fast, before anything is stored
        new_conversation = conversation_id is None
        conversation_id = conversation_id or uuid.uuid4()
        async with self._lock(conversation_id):
            # Transaction 1: the user message is stored even if the agent fails later.
            async with self._transaction() as session:
                conversation = await self._get_or_create(session, conversation_id, new_conversation, message)
                await repository.add_message(session, conversation, "user", message)

            try:
                values = await graph.ainvoke(
                    {"message": message}, {"configurable": {"thread_id": str(conversation_id)}}
                )
                output = output_from_state(values)
            except DATABASE_ERRORS as exc:  # checkpointer could not read/write
                raise DatabaseUnavailableError(f"Checkpoint storage failed: {type(exc).__name__}") from exc
            except Exception as exc:
                logger.exception("Fact-checking workflow failed for conversation %s", conversation_id)
                raise AgentFailedError("The fact-checking workflow failed", conversation_id) from exc

            # Transaction 2: assistant answer + verification, atomically.
            async with self._transaction() as session:
                conversation = await repository.get_conversation(session, conversation_id)
                if conversation is None:  # deleted while the agent was running
                    raise ConversationNotFoundError(str(conversation_id))
                await repository.add_message(
                    session, conversation, "assistant", output.answer,
                    decision=output.decision.value, claim=output.claim,
                    verification=output.verification.model_dump(mode="json") if output.verification else None,
                    limitations=output.limitations or None,
                )
        return conversation_id, output

    async def _get_or_create(
        self, session: AsyncSession, conversation_id: uuid.UUID, new: bool, message: str
    ) -> Conversation:
        if new:
            return await repository.create_conversation(session, message[:TITLE_CHARS], conversation_id)
        conversation = await repository.get_conversation(session, conversation_id)
        if conversation is None:  # never create a conversation from a client-chosen id
            raise ConversationNotFoundError(str(conversation_id))
        return conversation

    # --- Conversations ---------------------------------------------------------------

    async def list_conversations(self, limit: int = 50, offset: int = 0) -> list[tuple[Conversation, int]]:
        async with self._transaction() as session:
            return await repository.list_conversations(session, limit, offset)

    async def get_conversation(self, conversation_id: uuid.UUID) -> tuple[Conversation, list[Message]]:
        async with self._transaction() as session:
            conversation = await repository.get_conversation(session, conversation_id)
            if conversation is None:
                raise ConversationNotFoundError(str(conversation_id))
            return conversation, await repository.get_messages(session, conversation_id)

    async def delete_conversation(self, conversation_id: uuid.UUID) -> None:
        async with self._lock(conversation_id):
            async with self._transaction() as session:
                if not await repository.delete_conversation(session, conversation_id):
                    raise ConversationNotFoundError(str(conversation_id))
                # Inside the transaction: if the checkpoints cannot be deleted, the rows are
                # rolled back too, so history and agent memory never get out of sync.
                try:
                    await self.checkpointer.adelete_thread(str(conversation_id))
                except DATABASE_ERRORS as exc:
                    raise DatabaseUnavailableError("Could not delete checkpoints") from exc

    # --- Helpers ---------------------------------------------------------------------

    def _transaction(self) -> "_Transaction":
        return _Transaction(self.session_factory)


class _Transaction:
    """`async with` a session in a transaction; database errors become DatabaseUnavailableError."""

    def __init__(self, factory: async_sessionmaker[AsyncSession]) -> None:
        self.factory = factory

    async def __aenter__(self) -> AsyncSession:
        self.session = self.factory()
        try:
            await self.session.begin()
        except DATABASE_ERRORS as exc:
            await self.session.close()
            raise DatabaseUnavailableError(f"Database unavailable: {type(exc).__name__}") from exc
        return self.session

    async def __aexit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        try:
            if exc_type is None:
                await self.session.commit()
            else:
                await self.session.rollback()
        except DATABASE_ERRORS as db_exc:
            if exc_type is None:
                raise DatabaseUnavailableError(f"Database unavailable: {type(db_exc).__name__}") from db_exc
        finally:
            await self.session.close()
        if exc_type is not None and issubclass(exc_type, DATABASE_ERRORS):
            raise DatabaseUnavailableError(f"Database unavailable: {exc_type.__name__}") from exc


@asynccontextmanager
async def open_conversation_service(
    database_url: str,
    agent_factory: Callable[[], FactCheckingAgent] = FactCheckingAgent.from_settings,
) -> AsyncIterator[ConversationService]:
    """Engine + session factory for the app tables and a connection pool for the
    PostgreSQL checkpointer. Tables must exist already (scripts/init_db.py)."""
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    from psycopg.rows import dict_row
    from psycopg_pool import AsyncConnectionPool

    from app.database.session import checkpointer_conninfo, create_engine, create_session_factory
    from app.graph.workflow import make_serializer

    engine = create_engine(database_url)
    pool = AsyncConnectionPool(
        checkpointer_conninfo(database_url), open=False, max_size=10,
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
    )
    await pool.open(wait=False)  # do not block startup if PostgreSQL is down
    try:
        checkpointer = AsyncPostgresSaver(pool, serde=make_serializer())
        yield ConversationService(create_session_factory(engine), checkpointer, agent_factory)
    finally:
        await pool.close()
        await engine.dispose()
