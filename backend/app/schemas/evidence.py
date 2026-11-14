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
    stance: Stance
    relevance: float = Field(ge=0.0, le=1.0)
