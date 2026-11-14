"""Groq LLM integration. All Groq-specific code lives here."""

import json
from typing import Any

import groq
from groq import AsyncGroq
from pydantic import BaseModel, Field

from app.config import Settings, get_settings


class LLMError(Exception):
    """Base error for LLM failures. Callers should catch this, not groq errors."""


class LLMConfigurationError(LLMError):
    pass


class LLMTimeoutError(LLMError):
    pass


class LLMAPIError(LLMError):
    pass


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any]


class LLMMessage(BaseModel):
    """Assistant reply from a tool-enabled chat call."""

    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)

    def to_history(self) -> dict[str, Any]:
        """Assistant message in the chat format, to append to the conversation."""
        message: dict[str, Any] = {"role": "assistant", "content": self.content or ""}
        if self.tool_calls:
            message["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
                }
                for call in self.tool_calls
            ]
        return message


class LLMService:
    def __init__(self, settings: Settings | None = None, client: AsyncGroq | None = None) -> None:
        settings = settings or get_settings()
        if client is None:
            if settings.groq_api_key is None or not settings.groq_api_key.get_secret_value():
                raise LLMConfigurationError("GROQ_API_KEY is not set")
            client = AsyncGroq(
                api_key=settings.groq_api_key.get_secret_value(),
                timeout=settings.llm_timeout_seconds,
            )
        self._client = client
        self._model = settings.groq_model

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
    ) -> str:
        """Send a prompt and return the generated text, stripped."""
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = await self._client.chat.completions.create(
                model=self._model, messages=messages, temperature=temperature
            )
        except groq.APITimeoutError as exc:  # subclass of APIConnectionError: catch first
            raise LLMTimeoutError("LLM request timed out") from exc
        except groq.APIError as exc:
            raise LLMAPIError(f"LLM request failed: {exc}") from exc

        content = response.choices[0].message.content if response.choices else None
        if not content:
            raise LLMAPIError("LLM returned an empty response")
        return content.strip()

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
    ) -> LLMMessage:
        """Chat call with optional tool calling. The LLM may answer or request tool calls.

        `tools` uses the OpenAI-style function schema that Groq supports.
        Malformed tool-call arguments are reported as an empty dict; callers validate them.
        """
        kwargs: dict[str, Any] = {"model": self._model, "messages": messages, "temperature": temperature}
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        try:
            response = await self._client.chat.completions.create(**kwargs)
        except groq.APITimeoutError as exc:
            raise LLMTimeoutError("LLM request timed out") from exc
        except groq.APIError as exc:
            raise LLMAPIError(f"LLM request failed: {exc}") from exc

        if not response.choices:
            raise LLMAPIError("LLM returned no choices")
        message = response.choices[0].message
        calls = []
        for call in message.tool_calls or []:
            try:
                arguments = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {}
            calls.append(ToolCall(
                id=call.id,
                name=call.function.name,
                arguments=arguments if isinstance(arguments, dict) else {},
            ))
        return LLMMessage(content=message.content, tool_calls=calls)
