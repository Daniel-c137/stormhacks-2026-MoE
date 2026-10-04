"""Live: the standup and another meeting embedded by real Gemini, a window each, then searched.
Needs GEMINI_API_KEY, GEMINI_EMBEDDING_MODEL and GEMINI_EMBEDDING_DIM.
Deselected unless pytest runs with `-m live`."""

from pathlib import Path

import pytest

from brain.config import Settings
from brain.llm import GeminiEmbedder, make_embedder
from brain.memory import InMemoryMemoryStore, MeetingMemory
from brain.report import TranscriptInput
from contracts import TranscriptSegment

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


def other_meeting() -> list[TranscriptSegment]:
    """A meeting about something else, so the search has a wrong meeting to pass over."""
    lines = [
        ("p-alice", "Alice Moreau", "Let's plan the onboarding redesign for the mobile app."),
        ("p-carol", "Carol Jensen", "The new signup screens need copy and two illustrations."),
        ("p-alice", "Alice Moreau", "Then design reviews them on Thursday."),
    ]
    return [
        TranscriptSegment(
            seg_id=f"o-{n}",
            meeting_id="mtg-onboarding",
            speaker_id=speaker_id,
            speaker_name=name,
            text=text,
            is_final=True,
            t_start=n * 10,
            t_end=n * 10 + 8,
        )
        for n, (speaker_id, name, text) in enumerate(lines)
    ]


@pytest.mark.anyio
async def test_gemini_memory_finds_the_window_with_the_refund_turn():
    meeting = TranscriptInput.model_validate_json((FIXTURES / "standup.json").read_text())
    embedder = make_embedder(settings)
    assert isinstance(embedder, GeminiEmbedder)
    memory = MeetingMemory(embedder, InMemoryMemoryStore(dim=embedder.dim))

    await memory.index_meeting("t-live", meeting.meeting_id, meeting.segments)
    await memory.index_meeting("t-live", "mtg-onboarding", other_meeting())
    hits = await memory.search("t-live", "Who is handling refunds for the double charge?", k=3)

    for hit in hits:
        print(f"{hit.score:.3f} {hit.chunk.meeting_id} {hit.chunk.text!r}")
    top = hits[0].chunk
    assert top.meeting_id == meeting.meeting_id
    assert "Bob Okafor: The double-charge fix is merged." in top.text
    assert (top.t_start, top.t_end) == (0, 47)
    assert [h.chunk.meeting_id for h in hits] == [meeting.meeting_id, "mtg-onboarding"]
    assert hits[0].score > hits[1].score
