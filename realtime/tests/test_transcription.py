"""One transcription session per participant track: captions to the room, final segments to the
brain and the wake detector. One session failing never affects the others."""

import asyncio
from collections.abc import AsyncIterator, Callable

import pytest

from contracts import AGENT_PARTICIPANT_ID, Invocation, Topic, TranscriptSegment
from realtime_worker.brain_client import BrainUnavailable
from realtime_worker.invocation import WakeDetector
from realtime_worker.transcription import TranscriptionManager

pytestmark = pytest.mark.anyio

MEETING = "m-1"
AUDIO = object()  # stands in for the LiveKit audio frames; the fake STT ignores it
END = object()


class FakeSTT:
    """Each speaker's stream yields what the test makes them say, until ended or failed.
    Like the real one, it attaches the identity it was started with to every segment."""

    def __init__(self):
        self.queues: dict[str, asyncio.Queue] = {}
        self.started: list[tuple[str, str]] = []

    def _queue(self, speaker_id: str) -> asyncio.Queue:
        return self.queues.setdefault(speaker_id, asyncio.Queue())

    def say(self, speaker_id: str, text: str, *, final: bool = True) -> None:
        self._queue(speaker_id).put_nowait((text, final))

    def end(self, speaker_id: str) -> None:
        self._queue(speaker_id).put_nowait(END)

    def fail(self, speaker_id: str) -> None:
        self._queue(speaker_id).put_nowait(RuntimeError("Scribe connection dropped"))

    async def stream(
        self, meeting_id: str, speaker_id: str, speaker_name: str, audio
    ) -> AsyncIterator[TranscriptSegment]:
        assert audio is AUDIO
        self.started.append((speaker_id, speaker_name))
        queue = self._queue(speaker_id)
        n = 0
        while (item := await queue.get()) is not END:
            if isinstance(item, Exception):
                raise item
            text, final = item
            n += 1
            yield TranscriptSegment(
                seg_id=f"{meeting_id}-{speaker_id}-{n}",
                meeting_id=meeting_id,
                speaker_id=speaker_id,
                speaker_name=speaker_name,
                text=text,
                is_final=final,
                t_start=float(n),
                t_end=float(n) + 1,
            )


class FakeBus:
    def __init__(self):
        self.published: list[tuple[Topic, object, list[str] | None]] = []

    async def publish(self, topic, payload, *, to=None) -> None:
        self.published.append((topic, payload, to))

    def captions(self) -> list[str]:
        return [p.text for topic, p, _ in self.published if topic == Topic.TRANSCRIPT]


class FakeBrain:
    def __init__(self, failures: int = 0):
        self.failures = failures
        self.saved: list[tuple[str, TranscriptSegment]] = []

    async def ingest_segments(self, meeting_id: str, segments: list[TranscriptSegment]) -> None:
        if self.failures:
            self.failures -= 1
            raise BrainUnavailable("brain is down")
        self.saved.extend((meeting_id, s) for s in segments)

    def texts(self) -> list[str]:
        return [s.text for _, s in self.saved]


async def until(condition: Callable[[], bool], timeout: float = 1.0) -> None:
    async with asyncio.timeout(timeout):
        while not condition():
            await asyncio.sleep(0)


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
def invocations():
    return []


@pytest.fixture
async def manager(stt, bus, brain, invocations):
    async def on_invocation(invocation: Invocation) -> None:
        invocations.append(invocation)

    m = TranscriptionManager(
        MEETING,
        stt=stt,
        bus=bus,
        brain=brain,
        detector=WakeDetector(["OmniMan"]),
        on_invocation=on_invocation,
    )
    yield m
    await m.aclose()


# captions and saving


async def test_every_caption_reaches_the_room_and_only_finals_are_saved(manager, stt, bus, brain):
    manager.start("u-alex", "Alex Chen", "TR_alex_mic", AUDIO)
    stt.say("u-alex", "I'll finish", final=False)
    stt.say("u-alex", "I'll finish the auth API Friday.")
    stt.end("u-alex")

    await manager.join()

    assert bus.captions() == ["I'll finish", "I'll finish the auth API Friday."]
    assert all(to is None for _, _, to in bus.published)  # captions are for everyone
    assert brain.texts() == ["I'll finish the auth API Friday."]
    assert brain.saved[0][0] == MEETING


async def test_each_participant_is_transcribed_separately_under_their_own_identity(
    manager, stt, brain
):
    manager.start("u-alex", "Alex Chen", "TR_alex_mic", AUDIO)
    manager.start("u-sarah", "Sarah Kim", "TR_sarah_mic", AUDIO)
    stt.say("u-sarah", "Mobile depends on that API.")
    stt.say("u-alex", "Let's keep Postgres.")
    stt.end("u-alex")
    stt.end("u-sarah")

    await manager.join()

    assert sorted(stt.started) == [("u-alex", "Alex Chen"), ("u-sarah", "Sarah Kim")]
    by_speaker = {s.speaker_id: s.text for _, s in brain.saved}
    assert by_speaker == {
        "u-alex": "Let's keep Postgres.",
        "u-sarah": "Mobile depends on that API.",
    }


async def test_the_agents_own_audio_is_never_transcribed(manager, stt):
    manager.start(AGENT_PARTICIPANT_ID, "OmniMan", "TR_agent", AUDIO)

    assert stt.started == []
    assert manager.active_tracks() == set()


async def test_the_same_track_is_only_transcribed_once(manager, stt):
    manager.start("u-alex", "Alex Chen", "TR_alex_mic", AUDIO)
    manager.start("u-alex", "Alex Chen", "TR_alex_mic", AUDIO)
    await until(lambda: stt.started)
    await asyncio.sleep(0)

    assert stt.started == [("u-alex", "Alex Chen")]
    assert manager.active_tracks() == {"TR_alex_mic"}


# stopping


async def test_stopping_a_track_ends_only_that_session(manager, stt, brain):
    manager.start("u-alex", "Alex Chen", "TR_alex_mic", AUDIO)
    manager.start("u-sarah", "Sarah Kim", "TR_sarah_mic", AUDIO)

    await manager.stop("TR_alex_mic")  # Alex muted
    stt.say("u-alex", "this should never be heard")
    stt.say("u-sarah", "Still here.")
    stt.end("u-sarah")
    await manager.join()

    assert manager.active_tracks() == set()
    assert brain.texts() == ["Still here."]


async def test_a_participant_leaving_ends_all_their_sessions(manager, stt):
    manager.start("u-alex", "Alex Chen", "TR_alex_mic", AUDIO)
    manager.start("u-sarah", "Sarah Kim", "TR_sarah_mic", AUDIO)

    await manager.stop_participant("u-alex")

    assert manager.active_tracks() == {"TR_sarah_mic"}


async def test_a_participant_can_rejoin_after_leaving(manager, stt, brain):
    manager.start("u-sarah", "Sarah Kim", "TR_sarah_1", AUDIO)
    await manager.stop_participant("u-sarah")

    manager.start("u-sarah", "Sarah Kim", "TR_sarah_2", AUDIO)
    stt.say("u-sarah", "I'm back.")
    stt.end("u-sarah")
    await manager.join()

    assert brain.texts() == ["I'm back."]


async def test_close_ends_every_session(manager):
    manager.start("u-alex", "Alex Chen", "TR_alex_mic", AUDIO)
    manager.start("u-sarah", "Sarah Kim", "TR_sarah_mic", AUDIO)

    await manager.aclose()

    assert manager.active_tracks() == set()


# failures stay contained


async def test_one_speakers_transcription_failing_does_not_affect_the_others(manager, stt, brain):
    manager.start("u-alex", "Alex Chen", "TR_alex_mic", AUDIO)
    manager.start("u-sarah", "Sarah Kim", "TR_sarah_mic", AUDIO)

    stt.fail("u-alex")
    await until(lambda: manager.active_tracks() == {"TR_sarah_mic"})
    stt.say("u-sarah", "Did Alex drop?")
    stt.end("u-sarah")
    await manager.join()

    assert brain.texts() == ["Did Alex drop?"]


async def test_brain_being_down_does_not_stop_captions_or_later_saves(stt, bus, invocations):
    brain = FakeBrain(failures=1)

    async def on_invocation(invocation):
        invocations.append(invocation)

    manager = TranscriptionManager(
        MEETING,
        stt=stt,
        bus=bus,
        brain=brain,
        detector=WakeDetector(["OmniMan"]),
        on_invocation=on_invocation,
    )
    manager.start("u-alex", "Alex Chen", "TR_alex_mic", AUDIO)
    stt.say("u-alex", "Lost to the outage.")
    stt.say("u-alex", "Saved after it.")
    stt.end("u-alex")
    await manager.join()
    await manager.aclose()

    assert bus.captions() == ["Lost to the outage.", "Saved after it."]
    assert brain.texts() == ["Saved after it."]


# invocations


async def test_a_final_segment_naming_the_assistant_is_handed_on_as_an_invocation(
    manager, stt, invocations
):
    manager.start("u-alex", "Alex Chen", "TR_alex_mic", AUDIO)
    stt.say("u-alex", "OmniMan, what did we decide", final=False)
    stt.say("u-alex", "OmniMan, what did we decide about Postgres?")
    stt.say("u-alex", "Thanks.")
    stt.end("u-alex")
    await manager.join()

    assert [(i.asked_by_id, i.question) for i in invocations] == [
        ("u-alex", "what did we decide about Postgres?")
    ]
