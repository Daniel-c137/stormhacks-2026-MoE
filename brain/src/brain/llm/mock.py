import hashlib
import math
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from pydantic import BaseModel

from .base import Embeddings, EmbedTask, LLMError


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
        self.last_model: str | None = "mock"

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


class MockEmbedder:
    """Deterministic stand-in for tests and offline runs: a hashed bag of words, so texts that
    share words land close together. Its vectors are never presented as live embeddings."""

    model = "mock"

    def __init__(self, dim: int = 768):
        self.dim = dim
        self.calls: list[tuple[list[str], EmbedTask]] = []

    async def embed(self, texts: list[str], *, task: EmbedTask = "document") -> Embeddings:
        self.calls.append((list(texts), task))
        return Embeddings(model=self.model, vectors=[self._vector(t) for t in texts])

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for word in re.findall(r"[a-z0-9]+", text.casefold()):
            digest = hashlib.blake2b(word.encode(), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dim
            vector[bucket] += 1.0 if digest[4] & 1 else -1.0
        norm = math.sqrt(sum(x * x for x in vector)) or 1.0
        return [x / norm for x in vector]
