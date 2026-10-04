import logging
from collections.abc import Awaitable, Callable, Sequence

from pydantic import BaseModel

from .base import LLM, LLMOutOfCapacity

logger = logging.getLogger(__name__)


class FallbackLLM:
    """Tries `providers` in order, moving to the next only when one is out of capacity (every
    model overloaded, failing upstream or missing). Any other error is final. `last_model` is
    the answering provider's."""

    def __init__(self, providers: Sequence[LLM]):
        if not providers:
            raise ValueError("FallbackLLM needs at least one provider")
        self.providers = tuple(providers)
        self.last_model: str | None = None

    async def generate(self, prompt: str, *, system: str | None = None) -> str:
        return await self._first(lambda llm: llm.generate(prompt, system=system))

    async def generate_structured[T: BaseModel](
        self, prompt: str, schema: type[T], *, system: str | None = None
    ) -> T:
        return await self._first(lambda llm: llm.generate_structured(prompt, schema, system=system))

    async def _first[R](self, call: Callable[[LLM], Awaitable[R]]) -> R:
        self.last_model = None
        failures: list[str] = []
        last_error: LLMOutOfCapacity | None = None
        for n, llm in enumerate(self.providers):
            try:
                out = await call(llm)
            except LLMOutOfCapacity as e:
                failures.append(str(e))
                last_error = e
                if n + 1 < len(self.providers):
                    logger.warning("%s; trying the next model provider", e)
                continue
            self.last_model = llm.last_model
            return out
        raise LLMOutOfCapacity(" | ".join(failures)) from last_error
