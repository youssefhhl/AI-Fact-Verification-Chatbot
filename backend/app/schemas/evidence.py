from enum import StrEnum

from pydantic import BaseModel, Field, HttpUrl, field_serializer


class Stance(StrEnum):
    SUPPORTS = "SUPPORTS"
    CONTRADICTS = "CONTRADICTS"
    NEUTRAL = "NEUTRAL"


class Evidence(BaseModel):
    source: str = Field(min_length=1)
    url: HttpUrl | None = None
    content: str = Field(min_length=1)
    # None = not assessed yet. Set by the central Fact-Checking Agent, not by retrieval.
    stance: Stance | None = None
    # None when the source provides no score (e.g. Tavily MCP results).
    relevance: float | None = Field(default=None, ge=0.0, le=1.0)

    @field_serializer("url", when_used="always")
    def _url_as_str(self, url: HttpUrl | None) -> str | None:
        # Dump as plain str (also in python mode) so LangGraph checkpoints can store it.
        return str(url) if url is not None else None
