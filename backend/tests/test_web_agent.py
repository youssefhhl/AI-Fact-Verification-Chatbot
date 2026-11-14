from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.agents.web_agent import TOOLS, WebAgent, WebAgentError, to_evidence
from app.config import Settings
from app.schemas.research import ResearchRequest
from app.services.llm_service import LLMMessage, LLMTimeoutError, ToolCall
from app.services.tavily_mcp import TavilyConfigurationError, TavilyError, TavilyMCPClient, WebResult

pytestmark = pytest.mark.anyio

A = "https://en.wikipedia.org/wiki/Oliver_Reed"
B = "https://www.imdb.com/name/nm0001651/"
C = "https://example.org/blog"
RESULTS = [
    WebResult(url=A, title="Oliver Reed - Wikipedia", content="Reed was an English actor."),
    WebResult(url=B, title="Oliver Reed | IMDb", content="Actor known for Gladiator."),
]
REQUEST = ResearchRequest(claim="Oliver Reed was a film actor", instructions="Prefer encyclopedias")


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def call(name: str, i: int = 0, **arguments) -> ToolCall:
    return ToolCall(id=f"call_{name}_{i}", name=name, arguments=arguments)


def turn(*calls: ToolCall, content: str | None = None) -> LLMMessage:
    return LLMMessage(content=content, tool_calls=list(calls))


class FakeTools:
    def __init__(self, search=None, extract=None) -> None:
        self.search = AsyncMock(side_effect=search or (lambda *a, **k: list(RESULTS)))
        self.extract = AsyncMock(side_effect=extract or (
            lambda urls, query=None: [WebResult(url=u, content=f"Full text of {u}") for u in urls]
        ))


def make_agent(turns: list[LLMMessage], tools: FakeTools | None = None, **limits):
    llm = MagicMock()
    replies = iter(turns)
    llm.seen = []  # snapshot of the messages at each LLM call (the list is mutated later)

    async def chat(messages, tools=None):
        llm.seen.append([dict(m) for m in messages])
        reply = next(replies)
        if isinstance(reply, Exception):
            raise reply
        return reply

    llm.chat = AsyncMock(side_effect=chat)
    tools = tools or FakeTools()

    @asynccontextmanager
    async def factory():
        yield tools

    return WebAgent(llm, tools_factory=factory, **limits), llm, tools


def tool_names(llm_call) -> set[str]:
    return {t["function"]["name"] for t in llm_call.kwargs["tools"]}


async def test_search_extract_finish_flow() -> None:
    agent, llm, tools = make_agent([
        turn(call("web_search", query="Oliver Reed actor")),
        turn(call("extract_pages", urls=[A], focus="career")),
        turn(call("finish_research", relevant_urls=[A, B])),
    ])

    evidence = await agent.research(REQUEST)

    # The LLM chose the query and the tools; the controller executed them.
    tools.search.assert_awaited_once_with("Oliver Reed actor", time_range=None)
    tools.extract.assert_awaited_once_with([A], query="career")
    assert [str(e.url) for e in evidence] == [A, B]
    assert evidence[0].content == f"Full text of {A}"  # extracted content replaces snippet
    assert evidence[1].content == "Actor known for Gladiator."
    assert all(e.stance is None for e in evidence)
    # The request (claim + instructions) reached the LLM, and tool results were fed back.
    first_messages = llm.seen[0]
    assert "Oliver Reed was a film actor" in first_messages[1]["content"]
    assert "Prefer encyclopedias" in first_messages[1]["content"]
    final_messages = llm.seen[2]
    assert final_messages[-1]["role"] == "tool" and A in final_messages[-1]["content"]


async def test_refined_search_after_empty_results() -> None:
    results = iter([[], list(RESULTS)])
    tools = FakeTools(search=lambda *a, **k: next(results))
    agent, llm, _ = make_agent([
        turn(call("web_search", query="obscure phrasing")),
        turn(call("web_search", 1, query="Oliver Reed")),
        turn(call("finish_research", relevant_urls=[A])),
    ], tools)

    evidence = await agent.research(REQUEST)

    assert tools.search.await_count == 2
    second_turn_messages = llm.seen[1]
    assert "No results" in second_turn_messages[-1]["content"]
    assert [str(e.url) for e in evidence] == [A]


async def test_search_limit_enforced_by_controller() -> None:
    agent, llm, tools = make_agent([
        turn(call("web_search", 0, query="q0"), call("web_search", 1, query="q1")),
        turn(call("web_search", 2, query="q2")),
        turn(call("web_search", 3, query="q3")),  # LLM ignores the limit
        turn(content="done"),
    ], max_searches=2)

    await agent.research(REQUEST)

    assert tools.search.await_count == 2
    assert "web_search" not in tool_names(llm.chat.call_args_list[1])  # tool hidden once exhausted


async def test_iteration_limit_stops_loop() -> None:
    turns = [turn(call("web_search", i, query=f"q{i}")) for i in range(10)]
    agent, llm, _ = make_agent(turns, max_iterations=3, max_searches=10)

    evidence = await agent.research(REQUEST)

    assert llm.chat.await_count == 3
    assert {str(e.url) for e in evidence} == {A, B}  # fallback: search results


async def test_extraction_dedupes_and_respects_limit() -> None:
    agent, _, tools = make_agent([
        turn(call("web_search", query="q")),
        turn(call("extract_pages", urls=[A, A, C])),  # duplicate + URL not from search
        turn(call("extract_pages", 1, urls=[A, B])),  # A already extracted
        turn(content="done"),
    ], max_extractions=5)

    evidence = await agent.research(REQUEST)

    assert [c.args[0] for c in tools.extract.await_args_list] == [[A], [B]]
    assert {str(e.url) for e in evidence} == {A, B}  # fallback: extracted pages


async def test_extraction_budget_truncates_urls() -> None:
    agent, _, tools = make_agent([
        turn(call("web_search", query="q")),
        turn(call("extract_pages", urls=[A, B])),
        turn(content="done"),
    ], max_extractions=1)

    await agent.research(REQUEST)

    tools.extract.assert_awaited_once_with([A], query=None)


async def test_tool_errors_are_reported_to_llm_and_research_continues() -> None:
    tools = FakeTools(extract=lambda *a, **k: (_ for _ in ()).throw(TavilyError("timeout")))
    agent, llm, _ = make_agent([
        turn(call("web_search", query="q")),
        turn(call("extract_pages", urls=[A])),
        turn(call("finish_research", relevant_urls=[A])),
    ], tools)

    evidence = await agent.research(REQUEST)

    assert "Extraction failed" in llm.seen[2][-1]["content"]
    assert evidence[0].content == "Reed was an English actor."  # snippet kept


async def test_invalid_and_unknown_tool_calls() -> None:
    agent, llm, tools = make_agent([
        turn(call("web_search", query=""), call("delete_everything")),
        turn(content="done"),
    ])

    assert await agent.research(REQUEST) == []
    tools.search.assert_not_awaited()
    tool_messages = [m["content"] for m in llm.seen[1] if m["role"] == "tool"]
    assert tool_messages[0].startswith("Invalid arguments") and tool_messages[1].startswith("Unknown tool")


async def test_finish_ignores_unknown_urls() -> None:
    agent, _, _ = make_agent([
        turn(call("web_search", query="q")),
        turn(call("finish_research", relevant_urls=["https://made-up.example", B])),
    ])
    assert [str(e.url) for e in await agent.research(REQUEST)] == [B]


async def test_extract_tool_hidden_before_any_search() -> None:
    agent, llm, _ = make_agent([turn(content="nothing to do")])
    await agent.research(REQUEST)
    assert tool_names(llm.chat.call_args_list[0]) == {"web_search", "finish_research"}


async def test_llm_failure_before_research_raises() -> None:
    llm = MagicMock()
    llm.chat = AsyncMock(side_effect=LLMTimeoutError("t"))

    @asynccontextmanager
    async def factory():
        yield FakeTools()

    with pytest.raises(WebAgentError):
        await WebAgent(llm, tools_factory=factory).research(REQUEST)


async def test_llm_failure_mid_research_returns_partial_evidence() -> None:
    agent, _, _ = make_agent([turn(call("web_search", query="q")), LLMTimeoutError("t")])
    assert len(await agent.research(REQUEST)) == 2


async def test_missing_tavily_key_raises_agent_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        TavilyMCPClient, "from_settings",
        classmethod(lambda cls, s=None: cls("", "https://x")),
    )
    agent = WebAgent(MagicMock(), tools_factory=TavilyMCPClient.from_settings)
    with pytest.raises(WebAgentError, match="TAVILY_API_KEY"):
        await agent.research(REQUEST)


def test_missing_key_error_type() -> None:
    with pytest.raises(TavilyConfigurationError):
        TavilyMCPClient.from_settings(Settings(tavily_api_key=None))


def test_to_evidence() -> None:
    ev = to_evidence(WebResult(url=A, title="Oliver Reed", content="  text  ", score=1.3))
    assert ev.source == "en.wikipedia.org: Oliver Reed"
    assert ev.content == "text" and ev.relevance == 1.0 and ev.stance is None
    assert to_evidence(WebResult(url=B, content="x")).relevance is None


def test_tool_schemas_are_valid_function_definitions() -> None:
    for tool in TOOLS:
        assert tool["type"] == "function"
        assert tool["function"]["parameters"]["type"] == "object"
