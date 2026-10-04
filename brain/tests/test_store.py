"""Store guarantees the Supabase implementation must keep: ending is a single transition, and
joining appends a participant only if absent."""

import pytest

from brain.store import InMemoryStore
from contracts import Team

pytestmark = pytest.mark.anyio


@pytest.fixture
def store() -> InMemoryStore:
    return InMemoryStore(teams=[Team(id="t-1", name="Checkout", member_ids=["u-alex"])], people=[])


async def test_only_the_first_end_wins(store):
    meeting = await store.create_meeting("t-1", "Standup", host_id="u-alex")

    first = await store.end_meeting(meeting.id)
    second = await store.end_meeting(meeting.id)

    assert first is not None
    assert first.status == "processing"
    assert first.duration_min is not None
    assert second is None
    assert (await store.meeting(meeting.id)).status == "processing"


async def test_participant_is_appended_only_if_absent(store):
    meeting = await store.create_meeting("t-1", "Standup", host_id="u-alex")

    await store.add_participant(meeting.id, "u-alex")
    again = await store.add_participant(meeting.id, "u-alex")

    assert again.participant_ids == ["u-alex"]
