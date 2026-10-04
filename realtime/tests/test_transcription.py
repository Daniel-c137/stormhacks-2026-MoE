"""One transcription session per participant track: captions to the room, final segments to the
brain and the wake detector. The manager owns segment ids, the meeting clock and identity; the
speech-to-text only turns audio into text. Words are not lost to a dropped connection, a brain
outage, a mute or the meeting ending, and one session failing never affects the others."""

import asyncio
from collections.abc import Callable

import httpx
import pytest

from contracts import AGENT_PARTICIPANT_ID, Invocation, Topic, TranscriptSegment
from realtime_worker.brain_client import BrainRejected, BrainUnavailable, HttpBrainClient
from realtime_worker.invocation import WakeDetector
from realtime_worker.stt import SpeechPiece
from realtime_worker.transcription import TranscriptionManager

pytestmark = pytest.mark.anyio

MEETING = "m-1"
END = object()
AUDIO_ENDED = object()


class FakeAudio:
    """A participant's live microphone track: a frame every few milliseconds, forever, which a
    reopened stream can keep reading. Each fake frame is the speaker's id so FakeSTT can tell
    whose queue to read; the real transcriber never knows who is speaking."""

    def __init__(self, speaker_id: str):
        self.speaker_id = speaker_id

    def __aiter__(self):
        return self

    async def __anext__(self) -> str:
        await asyncio.sleep(0.002)
        return self.speaker_id


class FakeSTT:
    """Each speaker's stream yields what the test makes them say, until ended or failed.
    When its audio input ends it finalises the utterance in progress, like Scribe does."""

    def __init__(self):
        self.queues: dict[str, asyncio.Queue] = {}
        self.started: list[str] = []
        self.deaf_to_audio_end: set[str] = set()

    def _queue(self, speaker_id: str) -> asyncio.Queue:
        return self.queues.setdefault(speaker_id, asyncio.Queue())

    def say(self, speaker_id: str, text: str, *, final: bool = True) -> None:
        self._queue(speaker_id).put_nowait((text, final))

    def end(self, speaker_id: str) -> None:
        self._queue(speaker_id).put_nowait(END)

    def fail(self, speaker_id: str, times: int = 1) -> None:
        for _ in range(times):
            self._queue(speaker_id).put_nowait(RuntimeError("Scribe connection dropped"))

    async def stream(self, audio):
        try:
            speaker = await anext(audio)
        except StopAsyncIteration:
            return
        self.started.append(speaker)
        queue = self._queue(speaker)

        async def watch_audio():
            async for _ in audio:
                pass
            if speaker not in self.deaf_to_audio_end:
                queue.put_nowait(AUDIO_ENDED)

        watcher = asyncio.create_task(watch_audio())
        n, in_progress = 0, None
        try:
            while True:
                item = await queue.get()
                if item is END:
                    return
                if item is AUDIO_ENDED:
                    if in_progress:
                        yield SpeechPiece(text=in_progress, is_final=True, start=n, end=n + 1)
                    return
                if isinstance(item, Exception):
                    raise item
                text, final = item
                n += 1
                in_progress = None if final else text
                yield SpeechPiece(text=text, is_final=final, start=float(n), end=float(n) + 1)
        finally:
            watcher.cancel()


class FakeBus:
    def __init__(self):
        self.published: list[tuple[Topic, object, list[str] | None]] = []

    async def publish(self, topic, payload, *, to=None) -> None:
        self.published.append((topic, payload, to))

    def captions(self) -> list[TranscriptSegment]:
        return [p for topic, p, _ in self.published if topic == Topic.TRANSCRIPT]


class FakeBrain:
    """Saves like the real brain: idempotent by seg_id. `down` fails that many calls,
    `rejects` refuses any batch containing that text."""

    def __init__(self, down: int = 0, rejects: tuple[str, ...] = (), delay: float = 0):
        self.down = down
        self.rejects = rejects
        self.delay = delay
        self.saved: dict[str, TranscriptSegment] = {}

    async def ingest_segments(self, meeting_id: str, segments: list[TranscriptSegment]) -> None:
        await asyncio.sleep(self.delay)
        if self.down:
            self.down -= 1
            raise BrainUnavailable("brain is down")
        if any(s.text in self.rejects for s in segments):
            raise BrainRejected(422, "bad segment")
        for s in segments:
            self.saved.setdefault(s.seg_id, s)

    def texts(self) -> list[str]:
        return [s.text for s in sorted(self.saved.values(), key=lambda s: s.t_start)]


async def until(condition: Callable[[], bool], timeout: float = 1.0) -> None:
    async with asyncio.timeout(timeout):
        while not condition():
            await asyncio.sleep(0)


class Clock:
    """Seconds since the meeting started, set by the test."""

    def __init__(self, now: float = 0.0):
        self.now = now

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def stt():
    return FakeSTT()


@pytest.fixture
def bus():
    return FakeBus()


@pytest.fixture
def brain():
    return FakeBrain()


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def invocations():
    return []


@pytest.fixture
def unavailable():
    return []


@pytest.fixture
async def make_manager(stt, bus, clock, invocations, unavailable):
    managers = []

    def make(brain) -> TranscriptionManager:
        async def on_invocation(invocation: Invocation) -> None:
            invocations.append(invocation)

        async def on_unavailable(participant_id: str, name: str) -> None:
            unavailable.append(participant_id)

        m = TranscriptionManager(
            MEETING,
            stt=stt,
            bus=bus,
            brain=brain,
            detector=WakeDetector(["OmniMan"]),
            on_invocation=on_invocation,
            on_unavailable=on_unavailable,
            clock=clock,
            restart_backoff=0,
            drain_seconds=0.2,
        )
        managers.append(m)
        return m

    yield make
    for m in managers:
        await m.aclose()


@pytest.fixture
def manager(make_manager, brain):
    return make_manager(brain)


def start(manager: TranscriptionManager, speaker: str, track: str | None = None) -> None:
    names = {"u-alex": "Alex Chen", "u-sarah": "Sarah Kim", AGENT_PARTICIPANT_ID: "OmniMan"}
    manager.start(speaker, names[speaker], track or f"TR_{speaker}", FakeAudio(speaker))


# captions and saving


async def test_every_caption_reaches_the_room_and_only_finals_are_saved(manager, stt, bus, brain):
    start(manager, "u-alex")
    stt.say("u-alex", "I'll finish", final=False)
    stt.say("u-alex", "I'll finish the auth API Friday.")
    stt.end("u-alex")

    await manager.join()

    assert [c.text for c in bus.captions()] == ["I'll finish", "I'll finish the auth API Friday."]
    assert all(to is None for _, _, to in bus.published)  # captions are for everyone
    assert brain.texts() == ["I'll finish the auth API Friday."]


async def test_segments_carry_the_livekit_identity_of_the_track(manager, stt, brain):
    start(manager, "u-alex")
    start(manager, "u-sarah")
    stt.say("u-sarah", "Mobile depends on that API.")
    stt.say("u-alex", "Let's keep Postgres.")
    stt.end("u-alex")
    stt.end("u-sarah")

    await manager.join()

    assert sorted(stt.started) == ["u-alex", "u-sarah"]
    by_speaker = {(s.speaker_id, s.speaker_name): s.text for s in brain.saved.values()}
    assert by_speaker == {
        ("u-alex", "Alex Chen"): "Let's keep Postgres.",
        ("u-sarah", "Sarah Kim"): "Mobile depends on that API.",
    }
    assert all(s.meeting_id == MEETING for s in brain.saved.values())


async def test_the_agents_own_audio_is_never_transcribed(manager, stt):
    start(manager, AGENT_PARTICIPANT_ID)

    assert stt.started == []
    assert manager.active_tracks() == set()


async def test_the_same_track_is_only_transcribed_once(manager, stt):
    start(manager, "u-alex", "TR_alex_mic")
    start(manager, "u-alex", "TR_alex_mic")
    await until(lambda: stt.started)
    await asyncio.sleep(0)

    assert stt.started == ["u-alex"]
    assert manager.active_tracks() == {"TR_alex_mic"}


# segment ids and the meeting clock


async def test_a_partial_and_its_final_share_one_seg_id(manager, stt, bus):
    start(manager, "u-alex")
    stt.say("u-alex", "Let's keep", final=False)
    stt.say("u-alex", "Let's keep Postgres.")
    stt.say("u-alex", "Next item.")
    stt.end("u-alex")
    await manager.join()

    first_partial, first_final, second = bus.captions()
    assert first_partial.seg_id == first_final.seg_id
    assert second.seg_id != first_final.seg_id


async def test_times_are_seconds_from_the_meeting_start_not_the_stream_start(
    manager, stt, brain, clock
):
    clock.now = 100.0  # the stream opens 100 s into the meeting
    start(manager, "u-alex")
    stt.say("u-alex", "Late joiner here.")  # 1 s into its stream
    stt.end("u-alex")
    await manager.join()

    [saved] = brain.saved.values()
    assert (saved.t_start, saved.t_end) == (101.0, 102.0)


async def test_speaking_in_two_sessions_saves_both_utterances_in_the_real_brain(
    make_manager, stt, clock
):
    """Mute and unmute starts a new stream. Its ids must not collide with the first one's,
    or the brain's dedup would drop the new words."""
    from brain.api.deps import app_settings, get_store
    from brain.config import Settings as BrainSettings
    from brain.main import create_app
    from brain.store import InMemoryStore
    from contracts import Team

    token = "worker-shared-secret-0123456789abcdef"
    store = InMemoryStore(teams=[Team(id="t-1", name="T", member_ids=["u-sarah"])], people=[])
    meeting = await store.create_meeting("t-1", "Standup", host_id="u-sarah")
    await store.add_participant(meeting.id, "u-sarah")
    app = create_app()
    app.dependency_overrides[get_store] = lambda: store
    app.dependency_overrides[app_settings] = lambda: BrainSettings(
        _env_file=None, brain_internal_token=token
    )
    brain = HttpBrainClient("http://brain.test", token, transport=httpx.ASGITransport(app=app))
    manager = make_manager(brain)
    manager.meeting_id = meeting.id

    start(manager, "u-sarah", "TR_sarah_1")
    stt.say("u-sarah", "Before muting.")
    await until(lambda: manager.saved_count() == 1)
    await manager.stop("TR_sarah_1")
    clock.now = 30.0
    start(manager, "u-sarah", "TR_sarah_2")
    stt.say("u-sarah", "After unmuting.")
    stt.end("u-sarah")
    await manager.join()
    await manager.aclose()

    assert [s.text for s in await store.transcript(meeting.id)] == [
        "Before muting.",
        "After unmuting.",
    ]


# stopping without losing the last words


async def test_muting_lets_the_last_sentence_finish_before_stopping(manager, stt, brain):
    start(manager, "u-alex", "TR_alex_mic")
    stt.say("u-alex", "Okay, that's all from me", final=False)
    await until(lambda: len(stt.started) == 1)

    await manager.stop("TR_alex_mic")
    await manager.join()

    assert manager.active_tracks() == set()
    assert brain.texts() == ["Okay, that's all from me"]


async def test_stopping_gives_up_waiting_if_the_transcriber_never_finishes(manager, stt):
    stt.deaf_to_audio_end.add("u-alex")
    start(manager, "u-alex", "TR_alex_mic")
    await until(lambda: len(stt.started) == 1)

    async with asyncio.timeout(1):
        await manager.stop("TR_alex_mic")

    assert manager.active_tracks() == set()


async def test_stopping_a_track_ends_only_that_session(manager, stt, brain):
    start(manager, "u-alex", "TR_alex_mic")
    start(manager, "u-sarah", "TR_sarah_mic")

    await manager.stop("TR_alex_mic")
    stt.say("u-alex", "this should never be heard")
    stt.say("u-sarah", "Still here.")
    stt.end("u-sarah")
    await manager.join()

    assert manager.active_tracks() == set()
    assert brain.texts() == ["Still here."]


async def test_a_participant_leaving_ends_all_their_sessions(manager, stt):
    start(manager, "u-alex", "TR_alex_mic")
    start(manager, "u-sarah", "TR_sarah_mic")

    await manager.stop_participant("u-alex")

    assert manager.active_tracks() == {"TR_sarah_mic"}


async def test_closing_flushes_the_last_words_before_ending(make_manager, stt):
    brain = FakeBrain(delay=0.01)  # saves take a moment, as over the network
    manager = make_manager(brain)
    start(manager, "u-alex")
    stt.say("u-alex", "Meeting's over then.")
    stt.say("u-alex", "Bye everyone", final=False)
    await until(lambda: len(stt.started) == 1)

    await manager.aclose()

    assert brain.texts() == ["Meeting's over then.", "Bye everyone"]
    assert manager.active_tracks() == set()


# failures are retried, contained and counted


async def test_a_dropped_transcriber_connection_is_reopened(manager, stt, brain):
    start(manager, "u-alex")
    stt.fail("u-alex")
    stt.say("u-alex", "Still transcribed after the drop.")
    stt.end("u-alex")

    await manager.join()

    assert stt.started == ["u-alex", "u-alex"]
    assert brain.texts() == ["Still transcribed after the drop."]


async def test_after_repeated_drops_captions_are_reported_unavailable_and_others_continue(
    manager, stt, brain, unavailable
):
    start(manager, "u-alex", "TR_alex_mic")
    start(manager, "u-sarah", "TR_sarah_mic")

    stt.fail("u-alex", times=4)
    await until(lambda: manager.active_tracks() == {"TR_sarah_mic"})
    stt.say("u-sarah", "Did Alex drop?")
    stt.end("u-sarah")
    await manager.join()

    assert unavailable == ["u-alex"]
    assert brain.texts() == ["Did Alex drop?"]


async def test_segments_held_during_a_brain_outage_are_saved_once_it_is_back(
    make_manager, stt, bus
):
    brain = FakeBrain(down=1)
    manager = make_manager(brain)
    start(manager, "u-alex")
    stt.say("u-alex", "Said during the outage.")
    stt.say("u-alex", "Said after it.")
    stt.end("u-alex")
    await manager.join()
    await manager.aclose()

    assert [c.text for c in bus.captions()] == ["Said during the outage.", "Said after it."]
    assert brain.texts() == ["Said during the outage.", "Said after it."]
    assert manager.unsaved_count() == 0


async def test_segments_never_saved_are_counted(make_manager, stt):
    manager = make_manager(FakeBrain(down=1000))
    start(manager, "u-alex")
    stt.say("u-alex", "One.")
    stt.say("u-alex", "Two.")
    stt.end("u-alex")
    await manager.join()
    await manager.aclose()

    assert manager.unsaved_count() == 2


async def test_a_rejected_segment_does_not_block_the_ones_after_it(make_manager, stt):
    brain = FakeBrain(rejects=("Garbled.",))
    manager = make_manager(brain)
    start(manager, "u-alex")
    stt.say("u-alex", "Garbled.")
    stt.say("u-alex", "Clear.")
    stt.end("u-alex")
    await manager.join()
    await manager.aclose()

    assert brain.texts() == ["Clear."]
    assert manager.unsaved_count() == 1


# invocations


async def test_a_final_segment_addressing_the_assistant_is_handed_on(manager, stt, invocations):
    start(manager, "u-alex")
    stt.say("u-alex", "OmniMan, what did we decide", final=False)
    stt.say("u-alex", "OmniMan, what did we decide about Postgres?")
    stt.say("u-alex", "Thanks.")
    stt.end("u-alex")
    await manager.join()

    assert [(i.asked_by_id, i.question) for i in invocations] == [
        ("u-alex", "what did we decide about Postgres?")
    ]


async def test_ask_button_uses_the_meeting_clock(manager, stt, clock, invocations):
    start(manager, "u-sarah")
    clock.now = 0.5
    manager.arm_ask("u-sarah")
    stt.say("u-sarah", "Who owns the payment API?")  # 1 s into the meeting
    stt.end("u-sarah")
    await manager.join()

    assert [(i.via, i.question) for i in invocations] == [("ask", "Who owns the payment API?")]
