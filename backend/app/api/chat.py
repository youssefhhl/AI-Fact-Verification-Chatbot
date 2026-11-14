import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Query, Request, Response, status

from app.api.errors import ServiceNotConfiguredError
from app.database.models import Message
from app.schemas.chat import (
    ChatRequest,
    ChatResponse,
    ConversationDetail,
    ConversationSummary,
    ErrorResponse,
    MessageOut,
)
from app.services.conversation_service import ConversationService

router = APIRouter(prefix="/api", tags=["chat"])
ERRORS = {code: {"model": ErrorResponse} for code in (404, 422, 502, 503)}


def get_conversation_service(request: Request) -> ConversationService:
    service = getattr(request.app.state, "conversation_service", None)
    if service is None:
        raise ServiceNotConfiguredError()
    return service


Service = Annotated[ConversationService, Depends(get_conversation_service)]


@router.post("/chat", response_model=ChatResponse, responses=ERRORS)
async def chat(body: ChatRequest, service: Service) -> ChatResponse:
    conversation_id, output = await service.chat(body.message, body.conversation_id)
    return ChatResponse(
        conversation_id=conversation_id, answer=output.answer, verification=output.verification,
        decision=output.decision, limitations=output.limitations,
    )


@router.get("/conversations", response_model=list[ConversationSummary], responses=ERRORS)
async def list_conversations(
    service: Service,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[ConversationSummary]:
    rows = await service.list_conversations(limit, offset)
    return [
        ConversationSummary(id=c.id, title=c.title, created_at=c.created_at, updated_at=c.updated_at,
                            message_count=count)
        for c, count in rows
    ]


@router.get("/conversations/{conversation_id}", response_model=ConversationDetail, responses=ERRORS)
async def get_conversation(conversation_id: uuid.UUID, service: Service) -> ConversationDetail:
    conversation, messages = await service.get_conversation(conversation_id)
    return ConversationDetail(
        id=conversation.id, title=conversation.title, created_at=conversation.created_at,
        updated_at=conversation.updated_at, messages=[_message_out(m) for m in messages],
    )


@router.delete("/conversations/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT, responses=ERRORS)
async def delete_conversation(conversation_id: uuid.UUID, service: Service) -> Response:
    await service.delete_conversation(conversation_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


def _message_out(m: Message) -> MessageOut:
    return MessageOut(
        id=m.id, role=m.role, content=m.content, created_at=m.created_at, decision=m.decision,
        claim=m.claim, verification=m.verification, limitations=m.limitations or [],
    )
