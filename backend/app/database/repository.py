"""Persistence operations. Callers own the transaction (`async with session.begin()`)."""

import uuid
from typing import Any

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import Conversation, Message, utcnow


async def create_conversation(
    session: AsyncSession, title: str, conversation_id: uuid.UUID | None = None
) -> Conversation:
    conversation = Conversation(id=conversation_id or uuid.uuid4(), title=title[:200])
    session.add(conversation)
    await session.flush()
    return conversation


async def get_conversation(session: AsyncSession, conversation_id: uuid.UUID) -> Conversation | None:
    return await session.get(Conversation, conversation_id)


async def list_conversations(
    session: AsyncSession, limit: int = 50, offset: int = 0
) -> list[tuple[Conversation, int]]:
    counts = (
        select(Message.conversation_id, func.count(Message.id).label("n"))
        .group_by(Message.conversation_id).subquery()
    )
    rows = await session.execute(
        select(Conversation, func.coalesce(counts.c.n, 0))
        .outerjoin(counts, counts.c.conversation_id == Conversation.id)
        .order_by(Conversation.updated_at.desc()).limit(limit).offset(offset)
    )
    return [(conversation, count) for conversation, count in rows.all()]


async def get_messages(session: AsyncSession, conversation_id: uuid.UUID) -> list[Message]:
    rows = await session.scalars(
        select(Message).where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at, Message.role.desc())  # "user" before "assistant" on ties
    )
    return list(rows)


async def add_message(
    session: AsyncSession, conversation: Conversation, role: str, content: str, **fields: Any
) -> Message:
    message = Message(conversation_id=conversation.id, role=role, content=content, **fields)
    session.add(message)
    conversation.updated_at = utcnow()
    await session.flush()
    return message


async def delete_conversation(session: AsyncSession, conversation_id: uuid.UUID) -> bool:
    # Messages go with it (ON DELETE CASCADE).
    result = await session.execute(delete(Conversation).where(Conversation.id == conversation_id))
    return result.rowcount > 0
