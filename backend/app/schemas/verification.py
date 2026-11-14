from enum import StrEnum

from pydantic import BaseModel, Field

from app.schemas.evidence import Evidence


class Verdict(StrEnum):
    TRUE = "TRUE"
    FALSE = "FALSE"
    PARTIALLY_TRUE = "PARTIALLY_TRUE"
    NOT_ENOUGH_EVIDENCE = "NOT_ENOUGH_EVIDENCE"


class VerificationResult(BaseModel):
    claim: str = Field(min_length=1)
    verdict: Verdict
    explanation: str
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[Evidence] = Field(default_factory=list)
