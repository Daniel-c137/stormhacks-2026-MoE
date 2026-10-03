from typing import Protocol

from pydantic import BaseModel

from .agent.tools import ToolSpec


class LLM(Protocol):
    """Gemini behind a small interface; the model id comes from config."""

    async def generate(
        self, prompt: str, *, system: str | None = None, tools: list[ToolSpec] | None = None
    ) -> str: ...

    async def generate_structured[T: BaseModel](
        self, prompt: str, schema: type[T], *, system: str | None = None
    ) -> T: ...


class Embedder(Protocol):
    """Same model and dimensions for indexing and querying."""

    dim: int

    async def embed(self, texts: list[str]) -> list[list[float]]: ...
