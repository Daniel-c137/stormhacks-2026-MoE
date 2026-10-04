"""Meeting memory in Postgres with pgvector. The pool comes from the Postgres store; the table
from db/migrations/20261003000100_meeting_memory.sql."""

from psycopg import AsyncConnection
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from brain.llm import Embeddings

from .models import Chunk, MemoryHit, check_dim, check_vectors

DIM = 768  # vector(768) in the migration

COLUMNS = tuple(Chunk.model_fields)

UPSERT = f"""
insert into memory_chunks ({", ".join(COLUMNS)}, embedding_model, embedding)
values ({", ".join(["%s"] * len(COLUMNS))}, %s, %s::vector)
on conflict (id) do update set
    {", ".join(f"{c} = excluded.{c}" for c in COLUMNS[1:])},
    embedding_model = excluded.embedding_model,
    embedding = excluded.embedding
"""

SEARCH = f"""
select {", ".join(COLUMNS)}, 1 - (embedding <=> %(query)s::vector) as score
from memory_chunks
where team_id = %(team_id)s and embedding_model = %(model)s
order by embedding <=> %(query)s::vector, id
limit %(k)s
"""


class PgMemoryStore:
    def __init__(self, pool: AsyncConnectionPool, *, dim: int = DIM):
        self.pool = pool
        self.dim = dim

    async def index(self, chunks: list[Chunk], embeddings: Embeddings) -> None:
        check_vectors(self.dim, chunks, embeddings)
        async with self.pool.connection() as conn, conn.transaction():
            await self._insert(conn, chunks, embeddings)

    async def replace_meeting(
        self, meeting_id: str, chunks: list[Chunk], embeddings: Embeddings
    ) -> None:
        check_vectors(self.dim, chunks, embeddings)
        async with self.pool.connection() as conn, conn.transaction():
            await conn.execute("delete from memory_chunks where meeting_id = %s", [meeting_id])
            await self._insert(conn, chunks, embeddings)

    async def search(
        self, team_id: str, query: list[float], *, model: str, k: int = 8
    ) -> list[MemoryHit]:
        check_dim(self.dim, query)
        if k <= 0:
            return []
        params = {"query": literal(query), "team_id": team_id, "model": model, "k": k}
        async with self.pool.connection() as conn:
            cursor = await conn.cursor(row_factory=dict_row).execute(SEARCH, params)
            rows = await cursor.fetchall()
        return [MemoryHit(score=row.pop("score"), chunk=Chunk.model_validate(row)) for row in rows]

    async def delete_meeting_transcript(self, meeting_id: str) -> None:
        async with self.pool.connection() as conn:
            await conn.execute(
                "delete from memory_chunks where meeting_id = %s and kind = 'transcript'",
                [meeting_id],
            )

    async def _insert(
        self, conn: AsyncConnection, chunks: list[Chunk], embeddings: Embeddings
    ) -> None:
        if not chunks:
            return
        rows = [
            (*(getattr(chunk, c) for c in COLUMNS), embeddings.model, literal(vector))
            for chunk, vector in zip(chunks, embeddings.vectors, strict=True)
        ]
        async with conn.cursor() as cursor:
            await cursor.executemany(UPSERT, rows)


def literal(vector: list[float]) -> str:
    """pgvector's text form, so no client-side adapter is needed."""
    return "[" + ",".join(repr(float(x)) for x in vector) + "]"
