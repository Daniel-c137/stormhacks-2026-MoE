"""LiveKitRooms against a stand-in for LiveKit's room service: who is connected to a room."""

import asyncio

import pytest
from livekit.api import TwirpError
from livekit.protocol.models import ParticipantInfo
from livekit.protocol.room import ListParticipantsResponse

from brain import livekit_rooms
from brain.livekit_rooms import LiveKitRooms, NoRooms


class FakeLiveKitAPI:
    """Answers list_participants with `answer` (or raises it) and records being closed."""

    answer: ListParticipantsResponse | Exception = ListParticipantsResponse()
    closed = 0

    def __init__(self, *args):
        self.room = self

    async def list_participants(self, request):
        if isinstance(FakeLiveKitAPI.answer, Exception):
            raise FakeLiveKitAPI.answer
        return FakeLiveKitAPI.answer

    async def aclose(self):
        FakeLiveKitAPI.closed += 1


@pytest.fixture
def livekit(monkeypatch) -> type[FakeLiveKitAPI]:
    monkeypatch.setattr(livekit_rooms, "LiveKitAPI", FakeLiveKitAPI)
    FakeLiveKitAPI.answer = ListParticipantsResponse()
    FakeLiveKitAPI.closed = 0
    return FakeLiveKitAPI


def identities() -> list[str]:
    return asyncio.run(LiveKitRooms("wss://lk", "key", "secret").identities("m-1"))


def test_connected_participants_are_listed_and_disconnected_ones_are_not(livekit):
    state = ParticipantInfo.State
    livekit.answer = ListParticipantsResponse(
        participants=[
            ParticipantInfo(identity="u-alex", state=state.ACTIVE),
            ParticipantInfo(identity="u-sarah", state=state.DISCONNECTED),
            ParticipantInfo(identity="u-sam", state=state.JOINING),
        ]
    )

    assert identities() == ["u-alex", "u-sam"]
    assert livekit.closed == 1


def test_a_room_that_does_not_exist_has_nobody_in_it(livekit):
    livekit.answer = TwirpError("not_found", "room not found", status=404)

    assert identities() == []
    assert livekit.closed == 1


def test_other_livekit_errors_are_raised(livekit):
    livekit.answer = TwirpError("unavailable", "try later", status=503)

    with pytest.raises(TwirpError):
        identities()
    assert livekit.closed == 1


def test_without_livekit_nobody_is_connected():
    assert asyncio.run(NoRooms().identities("m-1")) == []
