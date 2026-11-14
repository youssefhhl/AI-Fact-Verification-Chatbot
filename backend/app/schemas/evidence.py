from enum import StrEnum

from pydantic import BaseModel, Field, HttpUrl


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
