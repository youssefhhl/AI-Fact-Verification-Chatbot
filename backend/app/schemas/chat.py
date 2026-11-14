import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from app.schemas.conversation import ContextDecision
from app.schemas.verification import VerificationResult


class ChatRequest(BaseModel):
    # strip_whitespace rejects messages that are only spaces.
    message: str = Field(min_length=1, max_length=4000)
    conversation_id: uuid.UUID | None = None  # omit to start a new conversation

    model_config = {"str_strip_whitespace": True}


class ChatResponse(BaseModel):
    conversation_id: uuid.UUID
    answer: str
    verification: VerificationResult | None = None
    decision: ContextDecision | None = None  # how the message was understood
    limitations: list[str] = Field(default_factory=list)  # research failures, if any


class ConversationSummary(BaseModel):
    id: uuid.UUID
    title: str
    created_at: datetime
    updated_at: datetime  # last activity
    message_count: int


class MessageOut(BaseModel):
    id: uuid.UUID
    role: str
    content: str
    created_at: datetime
    decision: ContextDecision | None = None
    claim: str | None = None
    verification: VerificationResult | None = None
    limitations: list[str] = Field(default_factory=list)


class ConversationDetail(BaseModel):
    id: uuid.UUID
    title: str
    created_at: datetime
    updated_at: datetime
    messages: list[MessageOut]


class ErrorBody(BaseModel):
    code: str
    message: str
    details: dict[str, Any] | None = None


class ErrorResponse(BaseModel):
    error: ErrorBody
