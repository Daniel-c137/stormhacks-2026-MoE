"""Live: the standup's turns embedded by real Gemini, then searched. Needs GEMINI_API_KEY,
GEMINI_EMBEDDING_MODEL and GEMINI_EMBEDDING_DIM. Deselected unless pytest runs with `-m live`."""

from pathlib import Path

import pytest

from brain.config import Settings
from brain.llm import GeminiEmbedder, make_embedder
from brain.memory import InMemoryMemoryStore, MeetingMemory
from brain.report import TranscriptInput

FIXTURES = Path(__file__).parent.parent / "fixtures"
settings = Settings()

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not (
            settings.gemini_api_key
            and settings.gemini_embedding_model
            and settings.gemini_embedding_dim
        ),
        reason="set GEMINI_API_KEY, GEMINI_EMBEDDING_MODEL and GEMINI_EMBEDDING_DIM",
    ),
]


@pytest.mark.anyio
async def test_gemini_memory_finds_the_refund_turn():
    meeting = TranscriptInput.model_validate_json((FIXTURES / "standup.json").read_text())
    embedder = make_embedder(settings)
    assert isinstance(embedder, GeminiEmbedder)
    memory = MeetingMemory(embedder, InMemoryMemoryStore(dim=embedder.dim))

    await memory.index_meeting("t-live", meeting.meeting_id, meeting.segments)
    hits = await memory.search("t-live", "Who is handling refunds for the double charge?", k=3)

    for hit in hits:
        print(f"{hit.score:.3f} {hit.chunk.text}")
    top = hits[0].chunk
    assert (top.speaker_id, top.t_start) == ("p-bob", 6)
    assert hits[0].score > hits[1].score
