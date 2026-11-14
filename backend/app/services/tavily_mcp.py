"""MCP client for Tavily's remote MCP server (https://mcp.tavily.com/mcp/).

Our backend is the MCP client; Tavily hosts the MCP server. We use two of its tools:
- tavily_search(query, ...)   -> list of results (title, url, content snippet)
- tavily_extract(urls, query) -> page content for given URLs

Tool names and arguments come from Tavily's official tavily-mcp server. The server
returns formatted text ("Title: ... / URL: ... / Content: ..."), which we parse here.
Authentication: the API key is passed as the `tavilyApiKey` query parameter (documented
method), so the URL contains a secret and must never be logged.
"""

import logging
from contextlib import AsyncExitStack
from types import TracebackType
from typing import Any, Self
from urllib.parse import urlencode

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from pydantic import BaseModel

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)
# The HTTP client logs request URLs at INFO, and our URL contains the API key.
logging.getLogger("httpx2").setLevel(logging.WARNING)


class TavilyError(Exception):
    """A Tavily MCP connection or tool call failed."""


class TavilyConfigurationError(TavilyError):
    """TAVILY_API_KEY is missing."""


class WebResult(BaseModel):
    url: str
    title: str = ""
    content: str = ""
    score: float | None = None  # only present if the server returns structured data


class TavilyMCPClient:
    """Async context manager holding one MCP session.

    Usage:
        async with TavilyMCPClient.from_settings() as tavily:
            results = await tavily.search("query")
    """

    def __init__(self, api_key: str, url: str = "https://mcp.tavily.com/mcp/") -> None:
        if not api_key:
            raise TavilyConfigurationError("TAVILY_API_KEY is not set")
        self._api_key = api_key
        self._url = f"{url}?{urlencode({'tavilyApiKey': api_key})}"
        self._stack: AsyncExitStack | None = None
        self._session: ClientSession | None = None

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "TavilyMCPClient":
        settings = settings or get_settings()
        key = settings.tavily_api_key.get_secret_value() if settings.tavily_api_key else ""
        return cls(key, settings.tavily_mcp_url)

    async def __aenter__(self) -> Self:
        self._stack = AsyncExitStack()
        try:
            streams = await self._stack.enter_async_context(streamable_http_client(self._url))
            session = ClientSession(streams[0], streams[1])
            self._session = await self._stack.enter_async_context(session)
            await self._session.initialize()
        except Exception as exc:
            await self._stack.aclose()
            raise TavilyError(f"Could not connect to Tavily MCP: {self._redact(exc)}") from None
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._stack:
            await self._stack.aclose()
        self._session = None

    async def search(self, query: str, max_results: int = 5, time_range: str | None = None) -> list[WebResult]:
        args: dict[str, Any] = {"query": query, "max_results": max_results}
        if time_range:
            args["time_range"] = time_range
        return await self._call("tavily_search", args)

    async def extract(self, urls: list[str], query: str | None = None) -> list[WebResult]:
        args: dict[str, Any] = {"urls": urls, "format": "text"}
        if query:
            args["query"] = query  # makes Tavily return only the chunks relevant to the query
        return await self._call("tavily_extract", args)

    async def _call(self, tool: str, args: dict[str, Any]) -> list[WebResult]:
        if self._session is None:
            raise TavilyError("Not connected; use `async with TavilyMCPClient(...)`")
        try:
            result = await self._session.call_tool(tool, args)
        except Exception as exc:
            raise TavilyError(f"{tool} failed: {self._redact(exc)}") from None
        text = "\n".join(getattr(c, "text", "") for c in result.content)
        if result.is_error:
            raise TavilyError(f"{tool} returned an error: {self._redact(text)[:300]}")
        if isinstance(result.structured_content, dict) and "results" in result.structured_content:
            return parse_structured(result.structured_content)
        return parse_results_text(text)

    def _redact(self, value: object) -> str:
        return str(value).replace(self._api_key, "***")


def parse_structured(data: dict[str, Any]) -> list[WebResult]:
    results = []
    for item in data.get("results", []):
        if not isinstance(item, dict) or not item.get("url"):
            continue
        results.append(WebResult(
            url=item["url"],
            title=item.get("title") or "",
            content=item.get("raw_content") or item.get("content") or "",
            score=item.get("score"),
        ))
    return results


def parse_results_text(text: str) -> list[WebResult]:
    """Parse tavily-mcp's text format (see its format-results.ts):

        Detailed Results:

        Title: ...
        URL: ...
        Content: ...            (search snippet)
        Raw Content: ...        (extract: may span many lines)
    """
    text = text.split("\nImages:", 1)[0]
    results: list[dict[str, str]] = []
    current: dict[str, str] = {}
    field: str | None = None
    keys = {"Title: ": "title", "URL: ": "url", "Content: ": "content",
            "Raw Content: ": "raw_content", "Favicon: ": None, "ID: ": None}

    for line in text.splitlines():
        prefix = next((p for p in keys if line.startswith(p)), None)
        if prefix is None:
            if field:  # continuation of a multi-line field
                current[field] += "\n" + line
            continue
        name = keys[prefix]
        # A new result starts at "Title:", or at "URL:" when the current one already has a URL.
        if name == "title" or (name == "url" and "url" in current):
            if current.get("url"):
                results.append(current)
            current = {}
        field = name
        if name:
            current[name] = line[len(prefix):]
    if current.get("url"):
        results.append(current)

    return [
        WebResult(
            url=r["url"].strip(),
            title=r.get("title", "").strip(),
            content=(r.get("raw_content") or r.get("content") or "").strip(),
        )
        for r in results
    ]
