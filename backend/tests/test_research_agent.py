from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.agents.research_agent import TOOLS, ResearchAgent, ResearchAgentError, to_evidence
from app.config import Settings
from app.schemas.research import ResearchRequest
from app.services.llm_service import LLMMessage, LLMTimeoutError, ToolCall
from app.services.scholarly_apis import (
    Publication,
    ScholarlyAPIError,
    ScholarlyRateLimitError,
    SearchFilters,
)

pytestmark = pytest.mark.anyio

REQUEST = ResearchRequest(claim="Vitamin D supplements prevent fractures in older adults")
REVIEW = Publication(
    provider="Europe PMC", provider_id="MED:1", title="Vitamin D and fractures: meta-analysis",
    abstract="No significant reduction in fractures.", year=2019, doi="10.1/abc",
    authors=["Lee H", "Kim J", "Park S", "Choi Y"], venue="BMJ",
)
SAME_PAPER_OPENALEX = REVIEW.model_copy(update={"provider": "OpenAlex", "provider_id": "W9", "doi": "10.1/ABC"})
TRIAL = Publication(provider="OpenAlex", provider_id="W2", title="Vitamin D RCT", abstract="Trial text.", year=2021)
NO_ABSTRACT = Publication(provider="Semantic Scholar", provider_id="s2x", title="Old study", url="https://www.semanticscholar.org/paper/s2x")


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def call(name: str, i: int = 0, **arguments) -> ToolCall:
    return ToolCall(id=f"{name}_{i}", name=name, arguments=arguments)


def turn(*calls: ToolCall, content: str | None = None) -> LLMMessage:
    return LLMMessage(content=content, tool_calls=list(calls))


def fake_tools(s2=None, openalex=None, epmc=None) -> MagicMock:
    tools = MagicMock()
    tools.search_semantic_scholar = AsyncMock(side_effect=s2 or (lambda *a, **k: [NO_ABSTRACT]))
    tools.search_openalex = AsyncMock(side_effect=openalex or (lambda *a, **k: [TRIAL]))
    tools.search_europe_pmc = AsyncMock(side_effect=epmc or (lambda *a, **k: [REVIEW]))
    return tools


def make_agent(turns, tools=None, **settings):
    replies = iter(turns)
    llm = MagicMock()
    llm.seen = []

    async def chat(messages, tools=None):
        llm.seen.append({"messages": [dict(m) for m in messages], "tools": {t["function"]["name"] for t in tools}})
        reply = next(replies)
        if isinstance(reply, Exception):
            raise reply
        return reply

    llm.chat = AsyncMock(side_effect=chat)
    tools = tools or fake_tools()

    @asynccontextmanager
    async def factory():
        yield tools

    return ResearchAgent(llm, tools_factory=factory, settings=Settings(**settings)), llm, tools


async def test_llm_selects_provider_and_filters() -> None:
    agent, llm, tools = make_agent([
        turn(call("search_europe_pmc", query="vitamin D supplementation fracture", filters={"reviews_only": True, "year_from": 2015})),
        turn(call("finish_research", relevant_ids=["doi:10.1/abc"])),
    ])

    evidence = await agent.research(REQUEST)

    tools.search_europe_pmc.assert_awaited_once_with(
        "vitamin D supplementation fracture", filters=SearchFilters(year_from=2015, reviews_only=True), limit=5
    )
    tools.search_openalex.assert_not_awaited()
    tools.search_semantic_scholar.assert_not_awaited()
    assert len(evidence) == 1 and evidence[0].stance is None
    assert str(evidence[0].url) == "https://doi.org/10.1/abc"
    assert REQUEST.claim in llm.seen[0]["messages"][1]["content"]
    assert "doi:10.1/abc" in llm.seen[1]["messages"][-1]["content"]  # results fed back to the LLM


async def test_multiple_providers_in_one_turn_and_cross_provider_dedup() -> None:
    tools = fake_tools(openalex=lambda *a, **k: [SAME_PAPER_OPENALEX, TRIAL])
    agent, llm, _ = make_agent([
        turn(call("search_europe_pmc", query="vitamin D fracture"),
             call("search_openalex", query="vitamin D fracture"),
             call("search_semantic_scholar", query="vitamin D fracture", fields_of_study=["Medicine"])),
        turn(content="enough"),  # no finish call -> all found publications
    ], tools)

    evidence = await agent.research(REQUEST)

    tools.search_semantic_scholar.assert_awaited_once_with(
        "vitamin D fracture", filters=SearchFilters(), limit=5, fields_of_study=["Medicine"]
    )
    # Same DOI from Europe PMC and OpenAlex counts once (case-insensitive DOI).
    assert len(evidence) == 3
    openalex_reply = llm.seen[1]["messages"][4]["content"]
    assert "doi:10.1/abc (already seen)" in openalex_reply


async def test_query_refinement_after_empty_results() -> None:
    results = iter([[], [REVIEW]])
    tools = fake_tools(epmc=lambda *a, **k: next(results))
    agent, llm, _ = make_agent([
        turn(call("search_europe_pmc", query="cholecalciferol osteoporotic fracture elderly nursing homes")),
        turn(call("search_europe_pmc", 1, query="vitamin D fracture")),
        turn(call("finish_research", relevant_ids=["doi:10.1/abc"])),
    ], tools)

    evidence = await agent.research(REQUEST)

    assert tools.search_europe_pmc.await_count == 2
    assert "No results" in llm.seen[1]["messages"][-1]["content"]
    assert len(evidence) == 1


async def test_duplicate_search_is_skipped() -> None:
    agent, llm, tools = make_agent([
        turn(call("search_openalex", query="Vitamin D  fracture")),
        turn(call("search_openalex", 1, query="vitamin d fracture")),  # same after normalization
        turn(content="done"),
    ])
    await agent.research(REQUEST)
    assert tools.search_openalex.await_count == 1
    assert "already ran this exact search" in llm.seen[2]["messages"][-1]["content"]


async def test_api_call_limit_enforced_and_tools_hidden() -> None:
    agent, llm, tools = make_agent([
        turn(*[call("search_openalex", i, query=f"query {i}") for i in range(4)]),  # asks for 4
        turn(call("search_europe_pmc", query="more")),  # LLM ignores the limit
        turn(content="done"),
    ], research_max_api_calls=2)

    await agent.research(REQUEST)

    assert tools.search_openalex.await_count == 2
    tools.search_europe_pmc.assert_not_awaited()
    assert llm.seen[1]["tools"] == {"finish_research"}


async def test_iteration_limit_stops_loop() -> None:
    turns = [turn(call("search_openalex", i, query=f"query {i}")) for i in range(10)]
    agent, llm, _ = make_agent(turns, research_max_iterations=3, research_max_api_calls=10)
    evidence = await agent.research(REQUEST)
    assert llm.chat.await_count == 3
    assert len(evidence) == 1  # TRIAL found (repeatedly), returned once


async def test_partial_provider_failure_continues_with_others() -> None:
    def rate_limited(*a, **k):
        raise ScholarlyRateLimitError("Semantic Scholar: rate limited (HTTP 429)")

    def broken(*a, **k):
        raise ScholarlyAPIError("OpenAlex: HTTP 500")

    tools = fake_tools(s2=rate_limited, openalex=broken)
    agent, llm, _ = make_agent([
        turn(call("search_semantic_scholar", query="vitamin D"), call("search_openalex", query="vitamin D")),
        turn(call("search_europe_pmc", query="vitamin D")),
        turn(call("finish_research", relevant_ids=["doi:10.1/abc"])),
    ], tools)

    evidence = await agent.research(REQUEST)

    assert [str(e.url) for e in evidence] == ["https://doi.org/10.1/abc"]
    feedback = [m["content"] for m in llm.seen[1]["messages"] if m["role"] == "tool"]
    assert "rate limited" in feedback[0] and "Search failed" in feedback[1]
    # Rate-limited provider is hidden afterwards; the one with a transient error is not.
    assert "search_semantic_scholar" not in llm.seen[1]["tools"]
    assert "search_openalex" in llm.seen[1]["tools"]


async def test_llm_failure_mid_research_returns_partial_evidence() -> None:
    agent, _, _ = make_agent([turn(call("search_europe_pmc", query="vitamin D")), LLMTimeoutError("t")])
    assert len(await agent.research(REQUEST)) == 1


async def test_llm_failure_before_any_result_raises() -> None:
    agent, _, _ = make_agent([LLMTimeoutError("t")])
    with pytest.raises(ResearchAgentError):
        await agent.research(REQUEST)


async def test_invalid_and_unknown_tool_calls() -> None:
    agent, llm, tools = make_agent([
        turn(call("search_openalex", query="x"),  # too short
             call("search_openalex", 1, query="vitamin D", filters={"year_from": 3000}),
             call("search_pubmed", query="vitamin D")),
        turn(content="done"),
    ])
    assert await agent.research(REQUEST) == []
    tools.search_openalex.assert_not_awaited()
    feedback = [m["content"] for m in llm.seen[1]["messages"] if m["role"] == "tool"]
    assert feedback[0].startswith("Invalid") and feedback[1].startswith("Invalid")
    assert feedback[2] == "Unknown tool: search_pubmed"


async def test_finish_ignores_unknown_ids() -> None:
    agent, _, _ = make_agent([
        turn(call("search_openalex", query="vitamin D")),
        turn(call("finish_research", relevant_ids=["doi:10.9/made-up", "OpenAlex:W2"])),
    ])
    assert [e.source for e in await agent.research(REQUEST)] == ["OpenAlex: Vitamin D RCT"]


def test_to_evidence_preserves_metadata_without_inventing() -> None:
    ev = to_evidence(REVIEW)
    assert ev.stance is None and ev.relevance is None
    assert ev.source == "Europe PMC: Vitamin D and fractures: meta-analysis"
    assert "(2019; Lee H, Kim J, Park S et al.; BMJ)" in ev.content
    assert ev.content.endswith("No significant reduction in fractures.")

    bare = to_evidence(NO_ABSTRACT)
    assert str(bare.url) == "https://www.semanticscholar.org/paper/s2x"  # no DOI -> provider URL
    assert "No abstract available from the provider." in bare.content

    tldr = to_evidence(NO_ABSTRACT.model_copy(update={"summary": "Short."}))
    assert "TLDR (machine-generated by Semantic Scholar): Short." in tldr.content
    assert to_evidence(TRIAL).url is None  # no DOI, no URL -> none invented


def test_tool_definitions() -> None:
    names = {t["function"]["name"] for t in TOOLS}
    assert names == {"search_semantic_scholar", "search_openalex", "search_europe_pmc", "finish_research"}
