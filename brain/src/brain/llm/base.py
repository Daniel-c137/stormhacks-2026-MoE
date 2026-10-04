from typing import Literal, Protocol

from pydantic import BaseModel


class LLMError(RuntimeError):
    """The model call failed or returned something unusable."""


class LLMUnavailable(LLMError):
    """The LLM is not configured. Never silently replaced by the mock."""


class LLM(Protocol):
    last_model: str | None
    """The model that answered the latest call, for logs and the CLI."""

    async def generate(self, prompt: str, *, system: str | None = None) -> str: ...

    async def generate_structured[T: BaseModel](
        self, prompt: str, schema: type[T], *, system: str | None = None
    ) -> T: ...


EmbedTask = Literal["document", "query"]


class Embeddings(BaseModel):
    """One vector per text, all from `model`. Vectors from different models are not comparable."""

    model: str
    vectors: list[list[float]]


class Embedder(Protocol):
    """Same model and dimensions for indexing and querying. `task` lets a provider embed stored
    documents and search queries differently."""

    dim: int

    async def embed(self, texts: list[str], *, task: EmbedTask = "document") -> Embeddings: ...
