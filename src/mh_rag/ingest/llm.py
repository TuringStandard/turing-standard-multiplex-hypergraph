"""Azure OpenAI structured-output client for Layer 2."""

from __future__ import annotations

import os
import time
from typing import Any, Protocol

import orjson
from openai import APIConnectionError, APIStatusError, AzureOpenAI, RateLimitError
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from mh_rag.config import Settings
from mh_rag.exceptions import ExtractionError


class LlmClient(Protocol):
    """Structured-output chat interface. FROZEN."""

    def complete_json(
        self, system: str, user: str, json_schema: dict, max_tokens: int
    ) -> dict:
        """Return a parsed JSON object from a schema-constrained completion."""
        ...


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, RateLimitError | APIConnectionError):
        return True
    if isinstance(exc, APIStatusError):
        return exc.status_code >= 500
    return False


class AzureLlmClient:
    """Azure OpenAI client with JSON-schema responses and client-side pacing."""

    def __init__(self, settings: Settings) -> None:
        """Bind Azure credentials and optional RPM throttle from the environment."""
        if not settings.azure_openai_api_key or not settings.azure_openai_endpoint:
            raise ExtractionError("AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY required")
        self._settings = settings
        self._client = AzureOpenAI(
            api_key=settings.azure_openai_api_key,
            api_version=settings.azure_openai_api_version,
            azure_endpoint=settings.azure_openai_endpoint,
        )
        self._deployment = settings.azure_openai_deployment
        rpm = float(os.environ.get("LLM_REQUESTS_PER_MINUTE", "20") or "20")
        self._min_interval = 60.0 / rpm if rpm > 0 else 0.0
        self._last_call_at = 0.0

    def complete_json(
        self, system: str, user: str, json_schema: dict, max_tokens: int
    ) -> dict:
        """Call chat.completions with strict JSON schema; retry rate limits / 5xx."""

        @retry(
            retry=retry_if_exception(_is_retryable),
            wait=wait_exponential(multiplier=2, min=2, max=30),
            stop=stop_after_attempt(3),
            reraise=True,
        )
        def _call() -> dict[str, Any]:
            self._throttle()
            response = self._client.chat.completions.create(
                model=self._deployment,
                temperature=0,
                seed=42,
                max_tokens=max_tokens,
                response_format={
                    "type": "json_schema",
                    "json_schema": json_schema,
                },
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            content = response.choices[0].message.content
            if not content:
                raise ExtractionError("LLM returned empty content")
            try:
                parsed = orjson.loads(content)
            except orjson.JSONDecodeError as exc:
                raise ExtractionError(f"LLM returned invalid JSON: {exc}") from exc
            if not isinstance(parsed, dict):
                raise ExtractionError("LLM JSON root must be an object")
            return parsed

        try:
            return _call()
        except ExtractionError:
            raise
        except Exception as exc:
            raise ExtractionError(f"LLM completion failed: {exc}") from exc

    def _throttle(self) -> None:
        if self._min_interval <= 0:
            return
        now = time.monotonic()
        wait = self._min_interval - (now - self._last_call_at)
        if wait > 0:
            time.sleep(wait)
        self._last_call_at = time.monotonic()
