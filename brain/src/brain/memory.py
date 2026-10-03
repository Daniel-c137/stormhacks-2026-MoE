from typing import Literal, Protocol

from pydantic import BaseModel

ChunkKind = Literal["transcript", "summary", "public_chat"]


class Chunk(BaseModel):
    """A pgvector row. Private chat is never embedded."""

    id: str
    team_id: str
    meeting_id: str
    kind: ChunkKind
    text: str
    speaker_id: str | None = None
    t_start: float | None = None
    t_end: float | None = None


class MemoryHit(BaseModel):
    chunk: Chunk
    score: float


class MemoryStore(Protocol):
    async def index(self, chunks: list[Chunk]) -> None: ...

    async def search(self, team_id: str, query: str, k: int = 8) -> list[MemoryHit]: ...
