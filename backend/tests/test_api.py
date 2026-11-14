"""Chat API end to end: FastAPI + ConversationService + LangGraph (in-memory
checkpointer) + SQLite for the application tables. Groq and research are faked."""

import uuid

import httpx
import pytest
from langgraph.checkpoint.memory import InMemorySaver

from app.api.chat import get_conversation_service
from app.config import Settings
from app.database import repository
from app.database.models import Base
from app.database.session import create_engine, create_session_factory
from app.graph.workflow import make_serializer
from app.main import create_app
from app.services.conversation_service import ConversationService
from app.services.llm_service import LLMConfigurationError
from tests.fakes import ELDERLY, FakeStack

pytestmark = pytest.mark.anyio
CLAIM = "Vitamin D prevents bone fractures"


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
async def session_factory(tmp_path):
    engine = create_engine(f"sqlite+aiosqlite:///{tmp_path / 'test.db'}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield create_session_factory(engine)
    await engine.dispose()


@pytest.fixture
def stack() -> FakeStack:
    return FakeStack()


@pytest.fixture
def service(session_factory, stack) -> ConversationService:
    return ConversationService(session_factory, InMemorySaver(serde=make_serializer()), stack.factory)


def client_for(service) -> httpx.AsyncClient:
    app = create_app(Settings(database_url=None))
    if service is not None:
        app.dependency_overrides[get_conversation_service] = lambda: service
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


@pytest.fixture
async def client(service):
    async with client_for(service) as c:
        yield c


async def chat(client, message, conversation_id=None, expected=200) -> dict:
    body = {"message": message} | ({"conversation_id": conversation_id} if conversation_id else {})
    response = await client.post("/api/chat", json=body)
    assert response.status_code == expected, response.text
    return response.json()


# --- Full chat flow ------------------------------------------------------------------

async def test_new_conversation_then_follow_ups_keep_context(client, stack) -> None:
    first = await chat(client, CLAIM)
    cid = first["conversation_id"]
    assert uuid.UUID(cid)
    assert first["decision"] == "NEW_VERIFICATION" and first["verification"]["verdict"] == "TRUE"
    assert {e["stance"] for e in first["verification"]["evidence"]} == {"SUPPORTS"}

    second = await chat(client, "What about elderly people?", cid)
    assert second["conversation_id"] == cid
    assert second["decision"] == "TARGETED_RESEARCH" and second["verification"]["claim"] == ELDERLY
    assert stack.research.requests[-1].claim == ELDERLY

    kb_calls = len(stack.retrieval.queries)
    third = await chat(client, "Can you explain the mechanism?", cid)
    assert third["decision"] == "REUSE_CONTEXT" and third["verification"] is None
    assert len(stack.retrieval.queries) == kb_calls  # no research for reuse

    detail = (await client.get(f"/api/conversations/{cid}")).json()
    assert [m["role"] for m in detail["messages"]] == ["user", "assistant"] * 3
    assert detail["messages"][0]["content"] == CLAIM
    assert detail["messages"][3]["claim"] == ELDERLY
    assert detail["messages"][3]["verification"]["verdict"] == "TRUE"
    assert detail["title"] == CLAIM


async def test_conversations_are_isolated(client, stack) -> None:
    a = (await chat(client, CLAIM))["conversation_id"]
    b_first = await chat(client, "Kubernetes is written in Go")
    assert b_first["conversation_id"] != a
    assert stack.llm.calls.count("context") == 0  # first message of each thread: no follow-up
    b_follow = await chat(client, "What about elderly people?", b_first["conversation_id"])
    # The follow-up resolves against conversation B's only topic, not A's vitamin D topic.
    assert b_follow["decision"] == "TARGETED_RESEARCH"
    detail_a = (await client.get(f"/api/conversations/{a}")).json()
    assert len(detail_a["messages"]) == 2


async def test_list_conversations_with_metadata(client) -> None:
    a = (await chat(client, CLAIM))["conversation_id"]
    b = (await chat(client, "Kubernetes is written in Go"))["conversation_id"]
    await chat(client, "What about elderly people?", a)  # a becomes most recent

    items = (await client.get("/api/conversations")).json()

    assert [i["id"] for i in items] == [a, b]
    assert items[0]["message_count"] == 4 and items[1]["message_count"] == 2
    assert items[0]["updated_at"] >= items[0]["created_at"]
    assert len((await client.get("/api/conversations", params={"limit": 1})).json()) == 1
    assert (await client.get("/api/conversations", params={"limit": 0})).status_code == 422


async def test_delete_conversation_removes_history_and_checkpoints(client, service) -> None:
    cid = (await chat(client, CLAIM))["conversation_id"]
    config = {"configurable": {"thread_id": cid}}
    assert await service.checkpointer.aget_tuple(config) is not None

    assert (await client.delete(f"/api/conversations/{cid}")).status_code == 204

    assert await service.checkpointer.aget_tuple(config) is None
    missing = await client.get(f"/api/conversations/{cid}")
    assert missing.status_code == 404 and missing.json()["error"]["code"] == "conversation_not_found"
    assert (await client.delete(f"/api/conversations/{cid}")).status_code == 404
    # Messages were deleted with the conversation (ON DELETE CASCADE).
    async with service.session_factory() as session:
        assert await repository.get_messages(session, uuid.UUID(cid)) == []


# --- Validation and errors -----------------------------------------------------------

async def test_unknown_conversation_id_is_rejected_not_created(client) -> None:
    unknown = str(uuid.uuid4())
    body = await chat(client, CLAIM, unknown, expected=404)
    assert body == {"error": {"code": "conversation_not_found", "message": "Conversation not found.",
                              "details": {"conversation_id": unknown}}}
    assert (await client.get("/api/conversations")).json() == []


@pytest.mark.parametrize("payload,field", [
    ({"message": "   "}, "message"),
    ({"message": "x" * 4001}, "message"),
    ({}, "message"),
    ({"message": "hi", "conversation_id": "not-a-uuid"}, "conversation_id"),
])
async def test_invalid_chat_requests(client, payload, field) -> None:
    response = await client.post("/api/chat", json=payload)
    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "validation_error" and error["details"]["errors"][0]["field"] == field


async def test_invalid_path_id_is_422(client) -> None:
    assert (await client.get("/api/conversations/123")).status_code == 422


async def test_missing_groq_key_returns_503_and_stores_nothing(session_factory) -> None:
    def no_key():
        raise LLMConfigurationError("GROQ_API_KEY is not set")

    service = ConversationService(session_factory, InMemorySaver(), no_key)
    async with client_for(service) as client:
        body = await chat(client, CLAIM, expected=503)
        assert body["error"]["code"] == "agent_unavailable"
        assert (await client.get("/api/conversations")).json() == []


async def test_agent_crash_returns_502_and_keeps_user_message(client, stack) -> None:
    async def crash(*args, **kwargs):
        raise RuntimeError("boom")

    stack.retrieval.search_evidence = crash  # a component failure is handled inside the turn
    ok = await chat(client, CLAIM)
    assert any("knowledge base failed" in x for x in ok["limitations"])
    assert ok["verification"]["verdict"] == "TRUE"  # the Research Agent still found evidence

    stack.llm.chat = crash  # not an LLMError: an unexpected bug makes the workflow fail
    body = await chat(client, "Kubernetes is written in Go", expected=502)
    cid = body["error"]["details"]["conversation_id"]
    messages = (await client.get(f"/api/conversations/{cid}")).json()["messages"]
    assert [m["role"] for m in messages] == ["user"]


async def test_database_unavailable_returns_503(tmp_path, stack) -> None:
    engine = create_engine(f"sqlite+aiosqlite:///{tmp_path / 'missing-dir' / 'x.db'}")
    service = ConversationService(create_session_factory(engine), InMemorySaver(), stack.factory)
    async with client_for(service) as client:
        assert (await chat(client, CLAIM, expected=503))["error"]["code"] == "database_unavailable"
        assert (await client.get("/api/conversations")).status_code == 503
    await engine.dispose()


async def test_failed_answer_persistence_is_rolled_back(client, monkeypatch) -> None:
    original = repository.add_message

    async def fail_for_assistant(session, conversation, role, content, **fields):
        message = await original(session, conversation, role, content, **fields)
        if role == "assistant":
            from sqlalchemy.exc import OperationalError
            raise OperationalError("INSERT", {}, Exception("disk full"))
        return message

    monkeypatch.setattr(repository, "add_message", fail_for_assistant)
    body = await chat(client, CLAIM, expected=503)
    assert body["error"]["code"] == "database_unavailable"
    monkeypatch.setattr(repository, "add_message", original)

    [conversation] = (await client.get("/api/conversations")).json()
    messages = (await client.get(f"/api/conversations/{conversation['id']}")).json()["messages"]
    assert [m["role"] for m in messages] == ["user"]  # assistant insert rolled back


async def test_chat_without_database_configured() -> None:
    async with client_for(None) as client:
        body = await chat(client, CLAIM, expected=503)
        assert body["error"]["code"] == "database_not_configured"
        assert (await client.get("/health")).status_code == 200
