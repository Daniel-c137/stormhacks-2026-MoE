import logging
from collections.abc import Sequence
from typing import Any

from google import genai
from google.genai import errors, types
from pydantic import BaseModel, ValidationError

from .base import Embeddings, EmbedTask, LLMError, LLMOutOfCapacity

# Overload and transient server errors: retried briefly on the same model, then the next model.
RETRYABLE_CODES = (429, 500, 502, 503, 504)
# A model that is retired or not offered to this key: not retried, but the next model may work.
# Anything else (bad request, auth, invalid schema) is a problem no other model will fix.
MISSING_MODEL_CODE = 404

logger = logging.getLogger(__name__)

# Models already reported missing in this process, so a stale config warns once, not per call.
warned_missing: set[str] = set()


def fail_fast_client(api_key: str | None, attempts: int, max_delay: float) -> genai.Client:
    """A client that retries overload briefly, so a fallback model gets its turn quickly."""
    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            retry_options=types.HttpRetryOptions(
                attempts=attempts,
                initial_delay=min(1.0, max_delay),
                max_delay=max_delay,
                http_status_codes=list(RETRYABLE_CODES),
            )
        ),
    )


def describe(model: str, e: errors.APIError) -> str:
    return f"{model}: {e.code} {e.status or ''}".rstrip() + f" ({e.message})"


def next_model_may_help(model: str, e: errors.APIError) -> bool:
    """Whether to try the next model after `e`; warns the first time `model` is missing."""
    if e.code == MISSING_MODEL_CODE:
        if model not in warned_missing:
            warned_missing.add(model)
            logger.warning(
                "Gemini model %s is not available to this key (%s); trying the next model. "
                "Update the configured model chain.",
                model,
                describe(model, e),
            )
        return True
    return e.code in RETRYABLE_CODES


class GeminiLLM:
    """Tries `models` in order (primary first), moving on only when a model is overloaded,
    failing transiently, or missing for this key. `last_model` is the model that answered the
    latest call."""

    def __init__(
        self,
        *,
        models: Sequence[str],
        api_key: str | None = None,
        client: Any = None,
        temperature: float | None = None,
        attempts: int = 2,
        max_delay: float = 4.0,
    ):
        if not models:
            raise ValueError("GeminiLLM needs at least one model")
        self.models = tuple(models)
        self.temperature = temperature
        self.last_model: str | None = None
        self._client = client or fail_fast_client(api_key, attempts, max_delay)

    async def generate(self, prompt: str, *, system: str | None = None) -> str:
        config = types.GenerateContentConfig(
            system_instruction=system, temperature=self.temperature
        )
        response = await self._generate(prompt, config)
        if not response.text:
            raise LLMError(f"Gemini ({self.last_model}) returned no content")
        return response.text

    async def generate_structured[T: BaseModel](
        self, prompt: str, schema: type[T], *, system: str | None = None
    ) -> T:
        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=self.temperature,
            response_mime_type="application/json",
            response_schema=schema,
        )
        response = await self._generate(prompt, config)
        if isinstance(response.parsed, schema):
            return response.parsed
        if not response.text:
            raise LLMError(f"Gemini ({self.last_model}) returned no content")
        try:
            return schema.model_validate_json(response.text)
        except ValidationError as e:
            raise LLMError(
                f"Gemini ({self.last_model}) response does not match {schema.__name__}: {e}"
            ) from e

    async def _generate(self, prompt: str, config: types.GenerateContentConfig) -> Any:
        self.last_model = None
        tried: list[str] = []
        last_error: errors.APIError | None = None
        for model in self.models:
            try:
                response = await self._client.aio.models.generate_content(
                    model=model, contents=prompt, config=config
                )
            except errors.APIError as e:
                if not next_model_may_help(model, e):
                    raise LLMError(f"Gemini request to {model} failed: {e}") from e
                tried.append(describe(model, e))
                last_error = e
                continue
            self.last_model = model
            return response
        raise LLMOutOfCapacity(
            f"Gemini is unavailable on every model tried: {'; '.join(tried)}"
        ) from last_error


EMBED_TASK_TYPES: dict[EmbedTask, str] = {
    "document": "RETRIEVAL_DOCUMENT",
    "query": "RETRIEVAL_QUERY",
}


class GeminiEmbedder:
    """Embeds with the first of `models` that is not overloaded or missing, at `dim` dimensions.
    Every vector of one call comes from one model: if a model fails part-way, the next model
    embeds all the texts again."""

    def __init__(
        self,
        *,
        models: Sequence[str],
        dim: int,
        api_key: str | None = None,
        client: Any = None,
        attempts: int = 2,
        max_delay: float = 4.0,
        batch_size: int = 100,  # the Gemini API's limit per batchEmbedContents request
    ):
        if not models:
            raise ValueError("GeminiEmbedder needs at least one model")
        self.models = tuple(models)
        self.dim = dim
        self.batch_size = batch_size
        self._client = client or fail_fast_client(api_key, attempts, max_delay)

    async def embed(self, texts: list[str], *, task: EmbedTask = "document") -> Embeddings:
        if not texts:
            return Embeddings(model=self.models[0], vectors=[])
        config = types.EmbedContentConfig(
            task_type=EMBED_TASK_TYPES[task], output_dimensionality=self.dim
        )
        tried: list[str] = []
        last_error: errors.APIError | None = None
        for model in self.models:
            vectors: list[list[float]] = []
            try:
                for start in range(0, len(texts), self.batch_size):
                    batch = texts[start : start + self.batch_size]
                    vectors += await self._embed_batch(model, batch, config)
            except errors.APIError as e:
                if not next_model_may_help(model, e):
                    raise LLMError(f"Gemini embedding request to {model} failed: {e}") from e
                tried.append(describe(model, e))
                last_error = e
                continue
            return Embeddings(model=model, vectors=vectors)
        raise LLMError(
            f"Gemini embeddings are unavailable on every model tried: {'; '.join(tried)}"
        ) from last_error

    async def _embed_batch(
        self, model: str, texts: list[str], config: types.EmbedContentConfig
    ) -> list[list[float]]:
        response = await self._client.aio.models.embed_content(
            model=model, contents=texts, config=config
        )
        vectors = [list(e.values or []) for e in response.embeddings or []]
        if len(vectors) != len(texts):
            raise LLMError(
                f"Gemini ({model}) returned {len(vectors)} vectors for {len(texts)} texts"
            )
        if any(len(v) != self.dim for v in vectors):
            raise LLMError(f"Gemini ({model}) returned vectors without {self.dim} dimensions")
        return vectors
