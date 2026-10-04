"""Helpers for tests of the post-meeting write-up."""

import asyncio

from pydantic import BaseModel

from brain.llm import MockLLM


class GatedLLM(MockLLM):
    """A MockLLM that holds every structured call until `gate` opens, so a test can look at a
    write-up while it runs. `called` is set once a call is waiting."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.gate = asyncio.Event()
        self.called = asyncio.Event()

    async def generate_structured[T: BaseModel](
        self, prompt: str, schema: type[T], *, system: str | None = None
    ) -> T:
        self.called.set()
        await self.gate.wait()
        return await super().generate_structured(prompt, schema, system=system)
