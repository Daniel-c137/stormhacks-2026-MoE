import math

from brain.llm import Embeddings

from .models import Chunk, MemoryHit, check_dim, check_vectors


class InMemoryMemoryStore:
    """Chunks and vectors in process memory with exact cosine search. For tests and local runs;
    everything is lost on restart."""

    def __init__(self, dim: int | None = None):
        self.dim = dim
        self._rows: dict[str, tuple[Chunk, str, list[float]]] = {}

    async def index(self, chunks: list[Chunk], embeddings: Embeddings) -> None:
        check_vectors(self.dim, chunks, embeddings)
        for chunk, vector in zip(chunks, embeddings.vectors, strict=True):
            self._rows[chunk.id] = (chunk.model_copy(), embeddings.model, vector)

    async def replace_meeting(
        self, meeting_id: str, chunks: list[Chunk], embeddings: Embeddings
    ) -> None:
        check_vectors(self.dim, chunks, embeddings)
        self._drop(lambda chunk: chunk.meeting_id == meeting_id)
        await self.index(chunks, embeddings)

    async def search(
        self, team_id: str, query: list[float], *, model: str, k: int = 8
    ) -> list[MemoryHit]:
        check_dim(self.dim, query)
        hits = [
            MemoryHit(chunk=chunk.model_copy(), score=cosine(query, vector))
            for chunk, row_model, vector in self._rows.values()
            if chunk.team_id == team_id and row_model == model
        ]
        hits.sort(key=lambda h: (-h.score, h.chunk.id))
        return hits[: max(k, 0)]

    async def delete_meeting_transcript(self, meeting_id: str) -> None:
        self._drop(lambda chunk: chunk.meeting_id == meeting_id and chunk.kind == "transcript")

    def _drop(self, matches) -> None:
        self._rows = {id: row for id, row in self._rows.items() if not matches(row[0])}


def cosine(a: list[float], b: list[float]) -> float:
    norms = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    if not norms:
        return 0.0
    return sum(x * y for x, y in zip(a, b, strict=True)) / norms
