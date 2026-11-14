"""Conversation state passed in and out of the Fact-Checking Agent on every turn.

The agent keeps no state of its own: the caller (later the API + PostgreSQL in
Phase 7) stores this object and sends it back with the next message.
"""

from enum import StrEnum

from pydantic import BaseModel, Field

from app.schemas.evidence import Evidence
from app.schemas.verification import Verdict, VerificationResult


class ContextDecision(StrEnum):
    NEW_VERIFICATION = "NEW_VERIFICATION"  # independent claim: full verification
    REUSE_CONTEXT = "REUSE_CONTEXT"  # follow-up answerable from existing evidence
    TARGETED_RESEARCH = "TARGETED_RESEARCH"  # follow-up that needs extra evidence


class Topic(BaseModel):
    """One claim the conversation verified, with everything collected for it."""

    claim: str
    # Evidence WITHOUT stance: stance depends on the exact claim it is judged against,
    # so it is re-assessed on every turn (e.g. "...in elderly people").
    evidence: list[Evidence] = Field(default_factory=list)
    verifications: list[VerificationResult] = Field(default_factory=list)
    evidence_gaps: list[str] = Field(default_factory=list)  # from the latest turn on this topic


class TurnRecord(BaseModel):
    message: str
    decision: ContextDecision
    topic: int  # index into ConversationState.topics
    claim: str  # standalone claim/question the turn was about
    answer: str
    verdict: Verdict | None = None


class ConversationState(BaseModel):
    turns: list[TurnRecord] = Field(default_factory=list)
    topics: list[Topic] = Field(default_factory=list)
    active_topic: int | None = None


class ResearchStep(BaseModel):
    """One research action taken during a turn (for transparency and debugging)."""

    tool: str  # "knowledge_base" | "research_agent" | "web_agent"
    query: str
    evidence_count: int = 0
    error: str | None = None


class FactCheckOutput(BaseModel):
    decision: ContextDecision
    claim: str
    answer: str  # user-facing reply
    verification: VerificationResult | None = None  # None for pure explanation follow-ups
    research_steps: list[ResearchStep] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)  # research failures, missing tools
    state: ConversationState
