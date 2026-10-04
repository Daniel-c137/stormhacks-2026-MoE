"""The worker's LiveKit glue: chat in and out over LiveKit's lk.chat text streams, and the meeting
clock segment times are on."""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import pytest
from livekit.rtc.data_stream import TextStreamInfo

from contracts import Meeting, TranslateResponse
from realtime_worker.config import Settings
from realtime_worker.worker import (
    CHAT_TOPIC,
    MeetingSession,
    RoomChat,
    meeting_clock,
    scribe_language,
    speech_translator,
)

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


# live translation (#106)


class TranslatingBrain:
    def __init__(self):
        self.calls: list[tuple[str, str, str | None]] = []

    async def translate(self, meeting_id: str, text: str, language: str | None):
        self.calls.append((meeting_id, text, language))
        return TranslateResponse(language="es", text="Hello")


def a_meeting(translate: bool) -> Meeting:
    return Meeting(
        id="m-1",
        team_id="t-1",
        title="Standup",
        status="live",
        code="abc",
        host_id="u-alex",
        participant_ids=[],
        translate=translate,
    )


async def test_a_meeting_with_translation_on_translates_through_the_brain():
    brain = TranslatingBrain()
    meeting = a_meeting(translate=True)

    translate = speech_translator(brain, meeting)
    answer = await translate("Hola", "es")

    assert answer == TranslateResponse(language="es", text="Hello")
    assert brain.calls == [("m-1", "Hola", "es")]
    assert scribe_language(Settings(_env_file=None), meeting) is None  # Scribe detects it


def test_a_meeting_with_translation_off_never_translates_and_stays_on_english():
    meeting = a_meeting(translate=False)

    assert speech_translator(TranslatingBrain(), meeting) is None
    assert scribe_language(Settings(_env_file=None), meeting) == "en"
    pinned = Settings(_env_file=None, elevenlabs_stt_language="fr")
    assert scribe_language(pinned, meeting) == "fr"


def test_a_sentence_still_going_waits_one_and_a_half_seconds_by_default():
    assert Settings(_env_file=None).translation_provisional_seconds == 1.5
