from pydantic import BaseModel, Field

from app.schemas.verification import VerificationResult


class ChatRequest(BaseModel):
    # strip_whitespace rejects messages that are only spaces.
    message: str = Field(min_length=1, max_length=4000)
    conversation_id: str | None = None

    model_config = {"str_strip_whitespace": True}


class ChatResponse(BaseModel):
    conversation_id: str
    answer: str
    verification: VerificationResult | None = None
