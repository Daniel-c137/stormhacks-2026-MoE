from collections.abc import Callable, Mapping
from dataclasses import dataclass

from pydantic import BaseModel

from .base import LLMError


@dataclass(frozen=True)
class LLMCall:
    prompt: str
    system: str | None
    schema: type[BaseModel] | None


class MockLLM:
    """Deterministic stand-in for tests and offline runs. Its output is never presented as live.

    Responses are scripted per schema, either as a fixed model or computed from the prompt.
    """

    def __init__(
        self,
        *,
        text: str | Callable[[str], str] | None = None,
        structured: Mapping[type[BaseModel], BaseModel | Callable[[str], BaseModel]] | None = None,
    ):
        self._text = text
        self._structured = dict(structured or {})
        self.calls: list[LLMCall] = []

    async def generate(self, prompt: str, *, system: str | None = None) -> str:
        self.calls.append(LLMCall(prompt, system, None))
        if self._text is None:
            raise LLMError("MockLLM has no scripted text response")
        return self._text(prompt) if callable(self._text) else self._text

    async def generate_structured[T: BaseModel](
        self, prompt: str, schema: type[T], *, system: str | None = None
    ) -> T:
        self.calls.append(LLMCall(prompt, system, schema))
        scripted = self._structured.get(schema)
        if scripted is None:
            raise LLMError(f"MockLLM has no scripted response for {schema.__name__}")
        out = scripted(prompt) if callable(scripted) else scripted
        return schema.model_validate(out.model_dump())
