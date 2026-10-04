"""OpenRouter's OpenAI-compatible chat completions and embeddings, the fallback when Gemini is
out of capacity.

Meeting transcripts go through it, so every request asks only for providers that keep no data
and support every parameter sent (structured output included)."""

import logging
import re
from collections.abc import Sequence
from typing import Any

import httpx
from pydantic import BaseModel, ValidationError

from .base import Embeddings, EmbedTask, LLMError, LLMOutOfCapacity

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


async def post(
    client: httpx.AsyncClient, endpoint: str, api_key: str, body: dict[str, Any]
) -> dict[str, Any]:
    """The JSON object OpenRouter answered with, or a RouteError with its status and message."""
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        response = await client.post(endpoint, json=body, headers=headers)
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
        return await post(client, self.endpoint, self._api_key, body)

    def _content(self, data: dict[str, Any]) -> str:
        choices = data.get("choices") or [{}]
        choice = choices[0] if isinstance(choices[0], dict) else {}
        if choice.get("finish_reason") == "length":
            raise LLMError(f"OpenRouter ({self.last_model}) stopped at max_tokens ({MAX_TOKENS})")
        content = (choice.get("message") or {}).get("content")
        if not isinstance(content, str) or not content.strip():
            raise LLMError(f"OpenRouter ({self.last_model}) returned no content")
        return content


# Gemini models on OpenRouter are "google/<Gemini name>".
GEMINI_PREFIX = "google/"
# OpenRouter's input types for Gemini's document and query tasks; a provider may ignore them.
EMBED_INPUT_TYPES: dict[EmbedTask, str] = {
    "document": "search_document",
    "query": "search_query",
}
EMBED_TIMEOUT = httpx.Timeout(30.0, connect=10.0)


def recorded_model_name(model: str) -> str:
    """The name OpenRouter's `model` is recorded under in memory: a Gemini model's own name, so
    its vectors are compared with those Gemini made; any other model keeps its OpenRouter id."""
    return model.removeprefix(GEMINI_PREFIX)


def without_input(message: str, texts: list[str]) -> str:
    """`message`, unless it quotes any of `texts` (four words in a row, or a whole short text)."""
    words = message.lower().split()
    quoted = {" ".join(words[i : i + 4]) for i in range(len(words))}
    flat = " ".join(words)
    for text in texts:
        tokens = text.lower().split()
        if len(tokens) < 4 and tokens and " ".join(tokens) in flat:
            return "message withheld: it quotes the input"
        if any(" ".join(tokens[i : i + 4]) in quoted for i in range(len(tokens) - 3)):
            return "message withheld: it quotes the input"
    return message


class OpenRouterEmbedder:
    """Embeds with one OpenRouter `model` at `dim` dimensions, recorded as `recorded_model`.
    Rate limits, timeouts and upstream failures are LLMOutOfCapacity; anything else is final.
    Texts are never logged or put in errors. `last_usage` is the latest call's usage."""

    def __init__(
        self,
        model: str,
        *,
        dim: int,
        api_key: str,
        base_url: str,
        transport: httpx.AsyncBaseTransport | None = None,
        batch_size: int = 100,
    ):
        self.model = model
        self.recorded_model = recorded_model_name(model)
        self.dim = dim
        self.batch_size = batch_size
        self.endpoint = f"{base_url.rstrip('/')}/embeddings"
        self.last_usage: dict[str, Any] | None = None
        self._api_key = api_key
        self._transport = transport

    async def embed(self, texts: list[str], *, task: EmbedTask = "document") -> Embeddings:
        self.last_usage = None
        vectors: list[list[float]] = []
        cost = 0.0
        async with httpx.AsyncClient(transport=self._transport, timeout=EMBED_TIMEOUT) as client:
            for start in range(0, len(texts), self.batch_size):
                batch = texts[start : start + self.batch_size]
                body = {
                    "model": self.model,
                    "input": batch,
                    "dimensions": self.dim,
                    "input_type": EMBED_INPUT_TYPES[task],
                    "encoding_format": "float",
                    "provider": PROVIDER_RULES,
                }
                try:
                    data = await post(client, self.endpoint, self._api_key, body)
                except RouteError as e:
                    failure = (
                        f"OpenRouter embedding request to {self.model} failed: {e.code} "
                        f"({without_input(e.message, batch)})"
                    )
                    if next_model_may_help(e.code):
                        raise LLMOutOfCapacity(failure) from None
                    raise LLMError(failure) from None
                vectors += self._vectors(data, len(batch))
                usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
                cost += float(usage.get("cost") or 0)
                self.last_usage = {"texts": len(texts), "cost": cost}
        return Embeddings(model=self.recorded_model, vectors=vectors)

    def _vectors(self, data: dict[str, Any], expected: int) -> list[list[float]]:
        items = data.get("data")
        items = [i for i in items if isinstance(i, dict)] if isinstance(items, list) else []
        if len(items) != expected:
            raise LLMError(
                f"OpenRouter ({self.model}) returned {len(items)} vectors for {expected} texts"
            )
        items.sort(key=lambda item: item.get("index", 0))
        vectors = [item.get("embedding") for item in items]
        if any(not isinstance(v, list) or len(v) != self.dim for v in vectors):
            raise LLMError(
                f"OpenRouter ({self.model}) returned vectors without {self.dim} dimensions"
            )
        return [[float(x) for x in v] for v in vectors]  # type: ignore[union-attr]
