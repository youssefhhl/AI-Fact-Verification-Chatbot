from unittest.mock import AsyncMock

import pytest
from mcp.types import CallToolResult, TextContent

from app.config import Settings
from app.services.tavily_mcp import (
    TavilyConfigurationError,
    TavilyError,
    TavilyMCPClient,
    parse_results_text,
    parse_structured,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


SEARCH_TEXT = """Detailed Results:

Title: Oliver Reed - Wikipedia
URL: https://en.wikipedia.org/wiki/Oliver_Reed
Content: Robert Oliver Reed was an English actor.
Favicon: https://en.wikipedia.org/favicon.ico

Title: Oliver Reed | IMDb
ID: abc
URL: https://www.imdb.com/name/nm0001651/
Content: Actor known for Gladiator.

Images:

[1] URL: https://img.example/1.png"""

EXTRACT_TEXT = """Detailed Results:

URL: https://en.wikipedia.org/wiki/Oliver_Reed
Raw Content: Line one.
Line two.

URL: https://example.org/page
Raw Content: Other page."""


def test_parse_search_text() -> None:
    results = parse_results_text(SEARCH_TEXT)
    assert [r.url for r in results] == [
        "https://en.wikipedia.org/wiki/Oliver_Reed", "https://www.imdb.com/name/nm0001651/",
    ]
    assert results[0].title == "Oliver Reed - Wikipedia"
    assert results[0].content == "Robert Oliver Reed was an English actor."
    assert results[0].score is None


def test_parse_extract_text_multiline_without_titles() -> None:
    results = parse_results_text(EXTRACT_TEXT)
    assert len(results) == 2
    assert results[0].content == "Line one.\nLine two."
    assert results[1].content == "Other page."


def test_parse_empty_text() -> None:
    assert parse_results_text("Detailed Results:\n") == []


def test_parse_structured_keeps_score() -> None:
    data = {"results": [{"url": "https://a.org", "title": "A", "content": "c", "score": 0.8}, {"title": "no url"}]}
    results = parse_structured(data)
    assert len(results) == 1 and results[0].score == 0.8


def test_missing_api_key() -> None:
    with pytest.raises(TavilyConfigurationError):
        TavilyMCPClient.from_settings(Settings(tavily_api_key=None))


def connected(result=None, error=None) -> TavilyMCPClient:
    client = TavilyMCPClient("secret-key")
    client._session = AsyncMock()
    client._session.call_tool = AsyncMock(return_value=result, side_effect=error)
    return client


async def test_search_calls_tavily_search_tool() -> None:
    client = connected(CallToolResult(content=[TextContent(type="text", text=SEARCH_TEXT)]))
    results = await client.search("Oliver Reed actor", time_range="year")
    client._session.call_tool.assert_awaited_once_with(
        "tavily_search", {"query": "Oliver Reed actor", "max_results": 5, "time_range": "year"}
    )
    assert len(results) == 2


async def test_extract_calls_tavily_extract_tool() -> None:
    client = connected(CallToolResult(content=[TextContent(type="text", text=EXTRACT_TEXT)]))
    await client.extract(["https://a.org"], query="actor")
    client._session.call_tool.assert_awaited_once_with(
        "tavily_extract", {"urls": ["https://a.org"], "format": "text", "query": "actor"}
    )


async def test_tool_error_result_raises() -> None:
    client = connected(CallToolResult(content=[TextContent(type="text", text="Invalid API key")], is_error=True))
    with pytest.raises(TavilyError):
        await client.search("x")


async def test_exceptions_are_wrapped_and_key_redacted() -> None:
    client = connected(error=RuntimeError("boom at https://mcp.tavily.com/mcp/?tavilyApiKey=secret-key"))
    with pytest.raises(TavilyError) as info:
        await client.search("x")
    assert "secret-key" not in str(info.value)


async def test_call_without_connection_raises() -> None:
    with pytest.raises(TavilyError):
        await TavilyMCPClient("k").search("x")
