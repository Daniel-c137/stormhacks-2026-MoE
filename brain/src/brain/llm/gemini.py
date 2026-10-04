from collections.abc import Sequence
from typing import Any

from google import genai
from google.genai import errors, types
from pydantic import BaseModel, ValidationError

from .base import LLMError

# Overload and transient server errors: retried briefly on the same model, then the next model.
# Anything else (bad request, auth, unknown model) is a problem no other model will fix.
RETRYABLE_CODES = (429, 500, 502, 503, 504)


class GeminiLLM:
    """Tries `models` in order (primary first), moving on only when a model is overloaded or
    failing transiently. `last_model` is the model that answered the latest call."""

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
        self._client = client or genai.Client(
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
                if e.code not in RETRYABLE_CODES:
                    raise LLMError(f"Gemini request to {model} failed: {e}") from e
                tried.append(f"{model}: {e.code} {e.status or ''}".rstrip() + f" ({e.message})")
                last_error = e
                continue
            self.last_model = model
            return response
        raise LLMError(
            f"Gemini is unavailable on every model tried: {'; '.join(tried)}"
        ) from last_error
