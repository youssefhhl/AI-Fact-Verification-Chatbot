from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import groq
import httpx
import pytest

from app.config import Settings
from app.services.llm_service import (
    LLMAPIError,
    LLMConfigurationError,
    LLMService,
    LLMTimeoutError,
)

pytestmark = pytest.mark.anyio

REQUEST = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def make_service(create: AsyncMock) -> LLMService:
    client = MagicMock()
    client.chat.completions.create = create
    return LLMService(settings=Settings(groq_model="test-model"), client=client)


def completion(text: str | None) -> SimpleNamespace:
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


async def test_generate_returns_text() -> None:
    create = AsyncMock(return_value=completion("  The claim is true.  "))
    service = make_service(create)

    answer = await service.generate("Is it true?", system_prompt="Be factual.")

    assert answer == "The claim is true."
    kwargs = create.call_args.kwargs
    assert kwargs["model"] == "test-model"
    assert kwargs["messages"][0] == {"role": "system", "content": "Be factual."}
    assert kwargs["messages"][1] == {"role": "user", "content": "Is it true?"}


def test_missing_api_key() -> None:
    with pytest.raises(LLMConfigurationError):
        LLMService(settings=Settings(groq_api_key=None))


def test_client_created_from_settings() -> None:
    service = LLMService(settings=Settings(groq_api_key="test-key", groq_model="m"))
    assert isinstance(service._client, groq.AsyncGroq)


async def test_api_error_is_wrapped() -> None:
    error = groq.InternalServerError(
        "boom", response=httpx.Response(500, request=REQUEST), body=None
    )
    service = make_service(AsyncMock(side_effect=error))
    with pytest.raises(LLMAPIError):
        await service.generate("hi")


async def test_timeout_is_wrapped() -> None:
    service = make_service(AsyncMock(side_effect=groq.APITimeoutError(request=REQUEST)))
    with pytest.raises(LLMTimeoutError):
        await service.generate("hi")


async def test_empty_response_raises() -> None:
    service = make_service(AsyncMock(return_value=completion("")))
    with pytest.raises(LLMAPIError):
        await service.generate("hi")


def tool_completion(content, tool_calls) -> SimpleNamespace:
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=tool_calls))])


async def test_chat_parses_tool_calls() -> None:
    raw_call = SimpleNamespace(
        id="c1", function=SimpleNamespace(name="web_search", arguments='{"query": "x"}')
    )
    bad_call = SimpleNamespace(id="c2", function=SimpleNamespace(name="web_search", arguments="{oops"))
    create = AsyncMock(return_value=tool_completion(None, [raw_call, bad_call]))
    service = make_service(create)
    tools = [{"type": "function", "function": {"name": "web_search", "parameters": {"type": "object"}}}]

    reply = await service.chat([{"role": "user", "content": "hi"}], tools=tools)

    assert create.call_args.kwargs["tools"] == tools
    assert create.call_args.kwargs["tool_choice"] == "auto"
    assert reply.tool_calls[0].name == "web_search" and reply.tool_calls[0].arguments == {"query": "x"}
    assert reply.tool_calls[1].arguments == {}  # malformed JSON -> empty, caller validates
    history = reply.to_history()
    assert history["role"] == "assistant" and history["tool_calls"][0]["id"] == "c1"


async def test_chat_without_tools_returns_text() -> None:
    create = AsyncMock(return_value=tool_completion("answer", None))
    reply = await make_service(create).chat([{"role": "user", "content": "hi"}])
    assert reply.content == "answer" and reply.tool_calls == []
    assert "tools" not in create.call_args.kwargs


async def test_chat_timeout_is_wrapped() -> None:
    service = make_service(AsyncMock(side_effect=groq.APITimeoutError(request=REQUEST)))
    with pytest.raises(LLMTimeoutError):
        await service.chat([{"role": "user", "content": "hi"}])
