"""The worker's LiveKit glue: chat in and out over LiveKit's lk.chat text streams, and the meeting
clock segment times are on."""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import pytest
from livekit.rtc.data_stream import TextStreamInfo

from contracts import Meeting
from realtime_worker.worker import CHAT_TOPIC, MeetingSession, RoomChat, meeting_clock

pytestmark = pytest.mark.anyio


def info(stream_id: str, timestamp_ms: int) -> TextStreamInfo:
    return TextStreamInfo(
        stream_id=stream_id,
        mime_type="text/plain",
        topic=CHAT_TOPIC,
        timestamp=timestamp_ms,
        size=None,
        attributes={},
        attachments=[],
    )


class Participant:
    def __init__(self):
        self.sent: list[tuple[str, str]] = []

    async def send_text(self, text: str, *, topic: str = "", **_) -> TextStreamInfo:
        self.sent.append((text, topic))
        return info("stream-9", 0)


async def test_polaris_posts_chat_on_livekits_chat_topic_and_gets_the_message_id():
    participant = Participant()

    message_id = await RoomChat(participant).send("The refund window is 14 days.")

    assert participant.sent == [("The refund window is 14 days.", "lk.chat")]
    assert message_id == "stream-9"


class Reader:
    def __init__(self, text: str, stream: TextStreamInfo):
        self._text = text
        self.info = stream

    async def read_all(self) -> str:
        return self._text


@dataclass
class Remote:
    name: str


class Room:
    remote_participants = {"u-alex": Remote("Alex Chen")}  # noqa: RUF012


class Agent:
    def __init__(self):
        self.chats = []

    async def on_chat(self, text, **message):
        self.chats.append((text, message))


async def test_incoming_chat_reaches_the_agent_with_the_livekit_sender_and_stream_id():
    agent = Agent()
    session = MeetingSession(Room(), None, None, agent, None, None)  # type: ignore[arg-type]
    sent_at = datetime(2026, 10, 3, 17, 5, tzinfo=UTC)

    await session._chat(
        Reader("@Polaris hi", info("stream-1", int(sent_at.timestamp() * 1000))), "u-alex"
    )

    assert agent.chats == [
        (
            "@Polaris hi",
            {
                "message_id": "stream-1",
                "sender_id": "u-alex",
                "sender_name": "Alex Chen",
                "ts": sent_at,
            },
        )
    ]


def test_the_clock_counts_from_the_meetings_start():
    started = datetime.now(UTC) - timedelta(seconds=90)
    meeting = Meeting(
        id="m-1",
        team_id="t-1",
        title="Standup",
        status="live",
        code="abc",
        host_id="u-alex",
        participant_ids=[],
        started_at=started,
    )

    assert 89 < meeting_clock(meeting)() < 92
