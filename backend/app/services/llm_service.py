"""Groq LLM integration. All Groq-specific code lives here."""

import groq
from groq import AsyncGroq

from app.config import Settings, get_settings


class LLMError(Exception):
    """Base error for LLM failures. Callers should catch this, not groq errors."""


class LLMConfigurationError(LLMError):
    pass


class LLMTimeoutError(LLMError):
    pass


class LLMAPIError(LLMError):
    pass


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
