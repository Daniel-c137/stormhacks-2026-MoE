from typing import Any

from google import genai
from google.genai import errors, types
from pydantic import BaseModel, ValidationError

from .base import LLMError


class GeminiLLM:
    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        client: Any = None,
        temperature: float | None = None,
    ):
        self.model = model
        self.temperature = temperature
        self._client = client or genai.Client(api_key=api_key)

    async def generate(self, prompt: str, *, system: str | None = None) -> str:
        config = types.GenerateContentConfig(
            system_instruction=system, temperature=self.temperature
        )
        response = await self._generate(prompt, config)
        if not response.text:
            raise LLMError("Gemini returned no content")
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
            raise LLMError("Gemini returned no content")
        try:
            return schema.model_validate_json(response.text)
        except ValidationError as e:
            raise LLMError(f"Gemini response does not match {schema.__name__}: {e}") from e

    async def _generate(self, prompt: str, config: types.GenerateContentConfig) -> Any:
        try:
            return await self._client.aio.models.generate_content(
                model=self.model, contents=prompt, config=config
            )
        except errors.APIError as e:
            raise LLMError(f"Gemini request failed: {e}") from e
