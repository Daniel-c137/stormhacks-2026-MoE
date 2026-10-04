"""OpenRouter's OpenAI-compatible chat completions, the fallback when Gemini is out of capacity.

Meeting transcripts go through it, so every request asks only for providers that keep no data
and support every parameter sent (structured output included)."""

import logging
import re
from collections.abc import Sequence
from typing import Any

import httpx
from pydantic import BaseModel, ValidationError

from .base import LLMError, LLMOutOfCapacity

# Bounds the cost of one call; a one-hour meeting's write-up fits well inside it.
MAX_TOKENS = 8192
TIMEOUT = httpx.Timeout(90.0, connect=10.0)
# Not offered (404), timed out (408), rate-limited (429) or failing upstream (5xx): the next
# model may answer. Anything else (bad request, auth, no credit) no other model will fix.
NEXT_MODEL_CODES = frozenset({404, 408, 429})
# Every request: only providers that honour all of its parameters and do not keep the data.
PROVIDER_RULES = {"require_parameters": True, "data_collection": "deny"}
# OpenAI-style strict mode needs every property required and no defaults; our schemas have
# optional fields, so the schema is a non-strict constraint and the answer is checked here.
STRICT = False

logger = logging.getLogger(__name__)
warned_missing: set[str] = set()

FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


class RouteError(Exception):
    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def next_model_may_help(code: int) -> bool:
    return code in NEXT_MODEL_CODES or code >= 500


def error_message(data: Any, fallback: str) -> str:
    error = data.get("error") if isinstance(data, dict) else None
    message = error.get("message") if isinstance(error, dict) else None
    text = " ".join(str(message or fallback).split())
    return text[:300]


def without_fence(text: str) -> str:
    match = FENCE.match(text.strip())
    return match.group(1) if match else text


class OpenRouterLLM:
    """Tries `models` (OpenRouter ids) in order, moving on only when a model is missing,
    rate-limited, timing out or failing upstream. `last_model` is "openrouter:<id>" for the
    model that answered the latest call; `last_usage` is that call's usage as OpenRouter reports
    it (tokens and cost). Prompts and answers are never logged."""

    def __init__(
        self,
        models: Sequence[str],
        *,
        api_key: str,
        base_url: str,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        if not models:
            raise ValueError("OpenRouterLLM needs at least one model")
        self.models = tuple(models)
        self.endpoint = f"{base_url.rstrip('/')}/chat/completions"
        self.last_model: str | None = None
        self.last_usage: dict[str, Any] | None = None
        self._api_key = api_key
        self._transport = transport

    async def generate(self, prompt: str, *, system: str | None = None) -> str:
        return await self._complete(prompt, system, None)

    async def generate_structured[T: BaseModel](
        self, prompt: str, schema: type[T], *, system: str | None = None
    ) -> T:
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": schema.__name__,
                "schema": schema.model_json_schema(),
                "strict": STRICT,
            },
        }
        text = await self._complete(prompt, system, response_format)
        try:
            return schema.model_validate_json(without_fence(text))
        except ValidationError as e:
            # The pydantic message quotes the answer; only where and what went wrong is kept.
            first = e.errors(include_input=False, include_url=False)[0]
            where = ".".join(str(part) for part in first["loc"]) or "the answer"
            raise LLMError(
                f"OpenRouter ({self.last_model}) response does not match {schema.__name__}: "
                f"{e.error_count()} problem(s), first at {where}: {first['msg']}"
            ) from None

    async def _complete(
        self, prompt: str, system: str | None, response_format: dict[str, Any] | None
    ) -> str:
        self.last_model = None
        self.last_usage = None
        messages = [{"role": "system", "content": system}] if system else []
        messages.append({"role": "user", "content": prompt})
        tried: list[str] = []
        async with httpx.AsyncClient(transport=self._transport, timeout=TIMEOUT) as client:
            for model in self.models:
                body: dict[str, Any] = {
                    "model": model,
                    "messages": messages,
                    "max_tokens": MAX_TOKENS,
                    "provider": PROVIDER_RULES,
                }
                if response_format:
                    body["response_format"] = response_format
                try:
                    data = await self._post(client, body)
                except RouteError as e:
                    if not next_model_may_help(e.code):
                        raise LLMError(
                            f"OpenRouter request to {model} failed: {e.code} ({e.message})"
                        ) from None
                    if e.code == 404 and model not in warned_missing:
                        warned_missing.add(model)
                        logger.warning(
                            "OpenRouter model %s is not available (%s); trying the next model. "
                            "Update OPENROUTER_MODELS.",
                            model,
                            e.message,
                        )
                    tried.append(f"{model}: {e.code} ({e.message})")
                    continue
                self.last_model = f"openrouter:{model}"
                self.last_usage = data.get("usage")
                return self._content(data)
        raise LLMOutOfCapacity(
            f"OpenRouter is unavailable on every model tried: {'; '.join(tried)}"
        )

    async def _post(self, client: httpx.AsyncClient, body: dict[str, Any]) -> dict[str, Any]:
        headers = {"Authorization": f"Bearer {self._api_key}"}
        try:
            response = await client.post(self.endpoint, json=body, headers=headers)
        except httpx.TimeoutException:
            raise RouteError(408, "timed out") from None
        except httpx.TransportError as e:
            raise RouteError(503, f"network error: {type(e).__name__}") from None
        try:
            data = response.json()
        except ValueError:
            data = None
        if response.is_error:
            raise RouteError(response.status_code, error_message(data, response.reason_phrase))
        if not isinstance(data, dict):
            raise RouteError(502, "the response is not a JSON object")
        if "error" in data:  # a provider's failure, passed on with a 200
            error = data["error"] if isinstance(data["error"], dict) else {}
            code = error.get("code")
            raise RouteError(code if isinstance(code, int) else 502, error_message(data, "error"))
        return data

    def _content(self, data: dict[str, Any]) -> str:
        choices = data.get("choices") or [{}]
        choice = choices[0] if isinstance(choices[0], dict) else {}
        if choice.get("finish_reason") == "length":
            raise LLMError(f"OpenRouter ({self.last_model}) stopped at max_tokens ({MAX_TOKENS})")
        content = (choice.get("message") or {}).get("content")
        if not isinstance(content, str) or not content.strip():
            raise LLMError(f"OpenRouter ({self.last_model}) returned no content")
        return content
