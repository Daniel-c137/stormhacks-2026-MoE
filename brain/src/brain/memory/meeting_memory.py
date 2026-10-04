from collections.abc import Iterable

from brain.llm import Embedder
from contracts import Report, TranscriptSegment

from .chunking import MAX_CHUNK_CHARS, chunk_report, chunk_transcript
from .models import Chunk, MemoryHit, MemoryStore


class MeetingMemory:
    """What the post-meeting pipeline indexes and the agent searches. Stored chunks are embedded
    as documents and questions as queries, with the same model and dimensions."""

    def __init__(
        self, embedder: Embedder, store: MemoryStore, *, max_chunk_chars: int = MAX_CHUNK_CHARS
    ):
        if store.dim is not None and store.dim != embedder.dim:
            raise ValueError(
                f"the embedder makes {embedder.dim}-dimension vectors; the store holds {store.dim}"
            )
        self.embedder = embedder
        self.store = store
        self.max_chunk_chars = max_chunk_chars

    async def index_meeting(
        self,
        team_id: str,
        meeting_id: str,
        segments: Iterable[TranscriptSegment],
        report: Report | None = None,
    ) -> list[Chunk]:
        """Replaces whatever memory the meeting had with its final transcript and its report."""
        if report is not None and report.meeting_id != meeting_id:
            raise ValueError(f"the report is for meeting {report.meeting_id}, not {meeting_id}")
        chunks = chunk_transcript(team_id, meeting_id, segments, max_chars=self.max_chunk_chars)
        if report is not None:
            chunks += chunk_report(team_id, report)
        embeddings = await self.embedder.embed([c.text for c in chunks], task="document")
        await self.store.replace_meeting(meeting_id, chunks, embeddings)
        return chunks

    async def search(self, team_id: str, query: str, k: int = 8) -> list[MemoryHit]:
        """The team's closest chunks, best first. A blank question finds nothing."""
        if not query.strip() or k <= 0:
            return []
        embedded = await self.embedder.embed([query], task="query")
        return await self.store.search(team_id, embedded.vectors[0], model=embedded.model, k=k)

    async def delete_meeting_transcript(self, meeting_id: str) -> None:
        await self.store.delete_meeting_transcript(meeting_id)
