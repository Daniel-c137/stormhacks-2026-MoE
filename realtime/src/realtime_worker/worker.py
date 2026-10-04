"""LiveKit agent worker. Joins each meeting's room as the agent (identity AGENT_PARTICIPANT_ID, the
agent's name from contracts/identity.json) with a published audio track for spoken answers.

Dispatch is automatic: LiveKit sends the worker a job for every new room, and only the brain issues
room tokens, each for a room named by its meeting id. Before connecting, the worker asks the brain
for that meeting and leaves any room that is not a live meeting.

Per participant microphone: Scribe -> captions on Topic.TRANSCRIPT -> final segments to the brain
-> the wake detector. Invocations (the name, the Ask button, a public @mention) go to the brain;
answers come back as a shared ResponseCard and are spoken only after a participant chooses Speak.
Public chat goes to the brain. Agenda and fact-check ticks run on timers. Participants joining and
leaving are followed, so a late joiner gets a private catch-up. When the room closes the job shuts
down, flushing the last final segments to the brain.
"""

import asyncio
import logging
import sys
import time
from collections.abc import AsyncIterator, Callable
from datetime import UTC, datetime

from livekit import rtc
from livekit.agents import AgentServer, AutoSubscribe, JobContext, JobRequest, cli

from contracts import (
    AGENT_PARTICIPANT_ID,
    Meeting,
    TranslateResponse,
    WorkerMeetingResponse,
    get_identity,
)

from .brain_client import BrainRejected, HttpBrainClient, brain_client_from_settings
from .bus import LiveKitBus
from .config import Settings, WorkerUnavailable, require
from .elevenlabs_tts import TTS_SAMPLE_RATE, ElevenLabsTTS
from .invocation import WakeDetector, default_aliases
from .meeting_agent import MeetingAgent
from .scribe import ScribeSTT
from .state import RoomAgentState
from .tracks import TrackRouter
from .transcription import TranscriptionManager, Translate

log = logging.getLogger(__name__)

# LiveKit's built-in chat (components-react useChat): a text stream on this topic.
CHAT_TOPIC = "lk.chat"


class TrackSpeaker:
    """The agent's published microphone track; silent until an answer is spoken."""

    def __init__(self, source: rtc.AudioSource):
        self._source = source

    @classmethod
    async def publish(cls, room: rtc.Room) -> "TrackSpeaker":
        source = rtc.AudioSource(TTS_SAMPLE_RATE, 1)
        track = rtc.LocalAudioTrack.create_audio_track("agent-voice", source)
        await room.local_participant.publish_track(
            track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
        )
        return cls(source)

    async def play(self, frames: AsyncIterator[rtc.AudioFrame]) -> None:
        try:
            async for frame in frames:
                await self._source.capture_frame(frame)
            await self._source.wait_for_playout()
        except BaseException:
            self._source.clear_queue()
            raise


class RoomChat:
    """Public room chat as the agent."""

    def __init__(self, participant: rtc.LocalParticipant):
        self._participant = participant

    async def send(self, text: str) -> str:
        info = await self._participant.send_text(text, topic=CHAT_TOPIC)
        return info.stream_id


def speech_translator(brain, meeting: Meeting) -> Translate | None:
    """translate(text, language) through the brain, only for a meeting whose host switched
    live translation on (#106); None otherwise, so nothing is ever sent to the model."""
    if not meeting.translate:
        return None

    async def translate(text: str, language: str | None) -> TranslateResponse:
        return await brain.translate(meeting.id, text, language)

    return translate


def scribe_language(settings: Settings, meeting: Meeting) -> str | None:
    """None lets Scribe detect each utterance's language, which translation needs; otherwise
    it is pinned (ELEVENLABS_STT_LANGUAGE, English by default), as without translation."""
    return None if meeting.translate else settings.elevenlabs_stt_language


# A caption in a meeting with translation on waits up to the brain client's translate timeout
# (4 s) for its translation before it is saved; the brain allows for it too (JEV_TRANSLATION_LAG_S).
TRANSLATION_LAG_S = 4.0


def agenda_check_delay(settings: Settings, meeting: Meeting) -> float:
    """How long after a caption ends the agenda is checked for it (with Jev keeping time)."""
    return settings.agenda_check_delay_seconds + (TRANSLATION_LAG_S if meeting.translate else 0)


def meeting_clock(meeting: Meeting) -> Callable[[], float]:
    """Seconds since the meeting started: the clock the brain keeps segment times on."""
    start = meeting.started_at.timestamp() if meeting.started_at else time.time()
    return lambda: max(0.0, time.time() - start)


class MeetingSession:
    """Everything the agent runs in one meeting room, wired to the room."""

    def __init__(
        self,
        room: rtc.Room,
        brain: HttpBrainClient,
        bus: LiveKitBus,
        agent: MeetingAgent,
        transcription: TranscriptionManager,
        router: TrackRouter,
    ):
        self.room = room
        self.brain = brain
        self.bus = bus
        self.agent = agent
        self.transcription = transcription
        self.router = router
        self._chat_tasks: set[asyncio.Task[None]] = set()

    @classmethod
    async def open(
        cls,
        room: rtc.Room,
        settings: Settings,
        brain: HttpBrainClient,
        info: WorkerMeetingResponse,
        keyterms: list[str],
    ) -> "MeetingSession":
        meeting = info.meeting
        name = get_identity().agent_name
        clock = meeting_clock(meeting)
        bus = LiveKitBus(room.local_participant)
        detector = WakeDetector(default_aliases(name))
        agent = MeetingAgent(
            meeting.id,
            bus=bus,
            brain=brain,
            state=RoomAgentState(bus),
            tts=ElevenLabsTTS(
                api_key=settings.elevenlabs_api_key or "",
                model=settings.elevenlabs_tts_model or "",
                url=settings.elevenlabs_api_url,
            ),
            speaker=await TrackSpeaker.publish(room),
            chat=RoomChat(room.local_participant),
            agent_name=name,
            default_voice_id=settings.elevenlabs_voice_id,
            detector=detector,
            spoken_max_chars=settings.spoken_answer_max_chars,
            agenda_tick_seconds=settings.agenda_tick_seconds,
            agenda_after_captions=settings.agenda_after_captions,
            agenda_check_delay=agenda_check_delay(settings, meeting),
            fact_check_tick_seconds=settings.fact_check_tick_seconds,
            clock=clock,
        )
        transcription = TranscriptionManager(
            meeting.id,
            stt=ScribeSTT(
                api_key=settings.elevenlabs_api_key or "",
                model=settings.elevenlabs_stt_model or "",
                keyterms=keyterms,
                language=scribe_language(settings, meeting),
                url=settings.elevenlabs_api_url,
                vad_silence_seconds=settings.elevenlabs_vad_silence_seconds,
            ),
            bus=bus,
            brain=brain,
            detector=detector,
            on_invocation=agent.on_invocation,
            on_listening=agent.on_listening,
            on_saved=lambda saved: agent.caption_saved(max(s.t_end for s in saved)),
            clock=clock,
            translate=speech_translator(brain, meeting),
            provisional_seconds=settings.translation_provisional_seconds,
            translation_pause_seconds=settings.translation_pause_seconds,
        )
        agent.transcription = transcription
        router = TrackRouter(transcription)
        session = cls(room, brain, bus, agent, transcription, router)

        bus.attach(room)
        router.attach(room)
        room.register_text_stream_handler(CHAT_TOPIC, session._on_chat)
        watch_presence(room, agent)
        for participant in room.remote_participants.values():  # subscribed before we listened
            for publication in participant.track_publications.values():
                if publication.subscribed and publication.track is not None:
                    router.subscribed(publication.track, publication, participant)
        agent.start()
        log.info("%s joined meeting %s", name, meeting.id)
        return session

    def _on_chat(self, reader: rtc.TextStreamReader, sender: str) -> None:
        task = asyncio.get_running_loop().create_task(self._chat(reader, sender))
        self._chat_tasks.add(task)
        task.add_done_callback(self._chat_tasks.discard)

    async def _chat(self, reader: rtc.TextStreamReader, sender: str) -> None:
        """sender is the identity LiveKit attached to the stream."""
        try:
            text = await reader.read_all()
            participant = self.room.remote_participants.get(sender)
            await self.agent.on_chat(
                text,
                message_id=reader.info.stream_id,
                sender_id=sender,
                sender_name=(participant.name if participant else "") or sender,
                ts=datetime.fromtimestamp(reader.info.timestamp / 1000, UTC),
            )
        except Exception:
            log.exception("Could not handle a chat message from %s", sender)

    async def aclose(self) -> None:
        """The room closed or the worker is stopping: let the last sentences finish and reach
        the brain, then stop everything."""
        await self.agent.aclose()
        await self.transcription.aclose()
        await self.router.aclose()
        for task in list(self._chat_tasks):
            task.cancel()
        await self.bus.aclose()
        if unsaved := self.transcription.unsaved_count():
            log.error(
                "%d final segment(s) of %s never reached the brain", unsaved, self.agent.meeting_id
            )
        await self.brain.aclose()


def watch_presence(room: rtc.Room, agent: MeetingAgent) -> None:
    """Who is already in the room, then everyone who joins or leaves after the agent did."""
    agent.already_here(list(room.remote_participants))
    room.on("participant_connected", lambda p: agent.on_participant_joined(p.identity))
    room.on("participant_disconnected", lambda p: agent.on_participant_left(p.identity))


async def on_request(request: JobRequest) -> None:
    await request.accept(identity=AGENT_PARTICIPANT_ID, name=get_identity().agent_name)


async def entrypoint(ctx: JobContext) -> None:
    settings = Settings()
    require(settings)
    meeting_id = ctx.room.name
    brain = brain_client_from_settings(settings)
    try:
        info = await brain.meeting(meeting_id)
    except BrainRejected as e:
        await brain.aclose()
        if e.status == 404:
            log.warning("Room %s is not a meeting; leaving it", meeting_id)
            ctx.shutdown("not a meeting")
            return
        raise
    if info.meeting.status != "live":
        await brain.aclose()
        log.warning("Meeting %s is %s; not joining", meeting_id, info.meeting.status)
        ctx.shutdown(f"meeting is {info.meeting.status}")
        return
    try:
        keyterms = await brain.keyterms(meeting_id)
    except Exception as e:
        log.warning("No keyterms for meeting %s, biasing only to the name: %s", meeting_id, e)
        keyterms = [get_identity().agent_name]

    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    session = await MeetingSession.open(ctx.room, settings, brain, info, keyterms)
    ctx.add_shutdown_callback(session.aclose)


def make_server(settings: Settings) -> AgentServer:
    """Automatic dispatch (no agent_name): every room the brain hands out a token for is a
    meeting, so the agent joins them all and leaves any that is not."""
    server = AgentServer(
        ws_url=settings.livekit_url or "",
        api_key=settings.livekit_api_key or "",
        api_secret=settings.livekit_api_secret or "",
    )
    server.rtc_session(entrypoint, on_request=on_request)
    return server


def main() -> None:
    settings = Settings()
    try:
        require(settings)
    except WorkerUnavailable as e:
        sys.exit(str(e))
    cli.run_app(make_server(settings))


if __name__ == "__main__":
    main()
