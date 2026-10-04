from typing import Literal, Protocol

from pydantic import BaseModel

from brain.llm import Embeddings

ChunkKind = Literal["transcript", "summary", "decision", "task", "public_chat"]


class Chunk(BaseModel):
    """A pgvector row. Private chat is never embedded.

    Transcript chunks are windows of consecutive turns: they carry the window's span, and its
    speaker only when one person speaks in all of it. Decision chunks carry who made the
    decision; decision and task chunks carry the report item's id in `ref_id` and its time, when
    known, in `t_start`.
    """

    id: str
    team_id: str
    meeting_id: str
    kind: ChunkKind
    text: str
    speaker_id: str | None = None
    speaker_name: str | None = None
    ref_id: str | None = None
    t_start: float | None = None
    t_end: float | None = None


class MemoryHit(BaseModel):
    chunk: Chunk
    score: float
    """Cosine similarity to the query, higher is closer."""


class MemoryStore(Protocol):
    """Chunks with their embeddings. Search is scoped to one team and only compares vectors
    from the same embedding model."""

    dim: int | None
    """The vector size the store holds, or None when any size goes."""

    async def index(self, chunks: list[Chunk], embeddings: Embeddings) -> None:
        """Adds chunks, replacing any with the same id."""
        ...

    async def replace_meeting(
        self, meeting_id: str, chunks: list[Chunk], embeddings: Embeddings
    ) -> None:
        """Atomically swaps everything stored for the meeting for these chunks."""
        ...

    async def search(
        self, team_id: str, query: list[float], *, model: str, k: int = 8
    ) -> list[MemoryHit]: ...

    async def delete_meeting_transcript(self, meeting_id: str) -> None:
        """Drops the meeting's transcript chunks; its report chunks stay."""
        ...


def check_vectors(dim: int | None, chunks: list[Chunk], embeddings: Embeddings) -> None:
    if len(chunks) != len(embeddings.vectors):
        raise ValueError(f"{len(chunks)} chunks but {len(embeddings.vectors)} vectors")
    for vector in embeddings.vectors:
        check_dim(dim, vector)


def check_dim(dim: int | None, vector: list[float]) -> None:
    if dim is not None and len(vector) != dim:
        raise ValueError(f"vector has {len(vector)} dimensions; the store holds {dim}")
