from pydantic import BaseModel, Field


class ResearchRequest(BaseModel):
    """What the central Fact-Checking Agent asks a research agent to investigate."""

    claim: str = Field(min_length=1)
    context: str | None = None  # e.g. conversation context or what is already known
    instructions: str | None = None  # e.g. "focus on recent news", "find official sources"

    model_config = {"str_strip_whitespace": True}
