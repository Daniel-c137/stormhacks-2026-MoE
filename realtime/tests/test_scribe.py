"""ElevenLabs Scribe v2 Realtime as the worker's SpeechToText: one websocket per track, partials
and finals as SpeechPieces, the last words finalised when the audio ends, and a dropped connection
raised so the TranscriptionManager reopens it."""

import asyncio
import base64
import json
from urllib.parse import parse_qs, urlsplit

import pytest
from livekit import rtc
from websockets.exceptions import ConnectionClosedError

from contracts import Invocation, TranscriptSegment
from realtime_worker.invocation import WakeDetector
from realtime_worker.scribe import SCRIBE_SAMPLE_RATE, ScribeError, ScribeSTT, parse_event
from realtime_worker.stt import SpeechPiece
from realtime_worker.transcription import TranscriptionManager

pytestmark = pytest.mark.anyio

KEY = "xi-test-key-never-logged"
END = object()


class FakeScribe:
    """The Scribe websocket: records what the worker sends; the test scripts the replies."""

    def __init__(self):
        self.url: str | None = None
        self.headers: dict[str, str] = {}
        self.sent: list[dict] = []
        self.inbox: asyncio.Queue = asyncio.Queue()
        self.closed = False
        self.on_commit = None

    async def send(self, text: str) -> None:
        message = json.loads(text)
        self.sent.append(message)
        if message.get("commit") and self.on_commit:
            self.on_commit()

    async def recv(self) -> str:
        item = await self.inbox.get()
        if isinstance(item, BaseException):
            raise item
        return item

    async def close(self) -> None:
        self.closed = True

    def reply(self, **message) -> None:
        self.inbox.put_nowait(json.dumps(message))

    def drop(self) -> None:
        self.inbox.put_nowait(ConnectionClosedError(None, None))

    def audio(self) -> list[dict]:
        return [m for m in self.sent if m["audio_base_64"]]

    def audio_bytes(self) -> int:
        return sum(len(base64.b64decode(m["audio_base_64"])) for m in self.audio())


class Connector:
    def __init__(self, *scribes: FakeScribe):
        self.scribes = list(scribes)
        self.opened: list[FakeScribe] = []

    async def __call__(self, url: str, headers: dict[str, str]) -> FakeScribe:
        scribe = self.scribes.pop(0)
        scribe.url, scribe.headers = url, headers
        self.opened.append(scribe)
        return scribe


class Mic:
    """A track's frames, as the test feeds them; end() ends the input."""

    def __init__(self):
        self.queue: asyncio.Queue = asyncio.Queue()

    def speak(self, seconds: float, *, rate: int = 16000, channels: int = 1) -> None:
        per_frame = rate // 100  # 10 ms
        for _ in range(round(seconds * 100)):
            self.queue.put_nowait(
                rtc.AudioFrame(
                    data=b"\x01\x00" * per_frame * channels,
                    sample_rate=rate,
                    num_channels=channels,
                    samples_per_channel=per_frame,
                )
            )

    def end(self) -> None:
        self.queue.put_nowait(END)

    def __aiter__(self):
        return self

    async def __anext__(self) -> rtc.AudioFrame:
        item = await self.queue.get()
        if item is END:
            raise StopAsyncIteration
        return item


def stt(connector: Connector, **options) -> ScribeSTT:
    return ScribeSTT(
        api_key=KEY,
        model="scribe_v2_realtime",
        keyterms=["Polaris", "DS-104", "refund window"],
        connect=connector,
        **{"finalize_seconds": 0.2, **options},
    )


def words(*timed: tuple[str, float, float]) -> list[dict]:
    out = []
    for text, start, end in timed:
        if out:
            out.append({"text": " ", "start": start, "end": start, "type": "spacing"})
        out.append({"text": text, "start": start, "end": end, "type": "word", "logprob": -0.1})
    return out


async def collect(stream, into: list) -> None:
    async for piece in stream:
        into.append(piece)


async def until(condition, timeout: float = 2.0) -> None:
    async with asyncio.timeout(timeout):
        while not condition():
            await asyncio.sleep(0.005)


# the connection


async def test_opens_one_realtime_session_with_the_model_format_vad_and_keyterms():
    scribe = FakeScribe()
    mic = Mic()
    mic.end()

    await collect(stt(Connector(scribe)).stream(mic), [])

    url = urlsplit(scribe.url)
    assert (url.scheme, url.netloc, url.path) == (
        "wss",
        "api.elevenlabs.io",
        "/v1/speech-to-text/realtime",
    )
    query = parse_qs(url.query)
    assert query["model_id"] == ["scribe_v2_realtime"]
    assert query["audio_format"] == [f"pcm_{SCRIBE_SAMPLE_RATE}"]
    assert query["commit_strategy"] == ["vad"]
    assert query["include_timestamps"] == ["true"]
    assert query["keyterms"] == ["Polaris", "DS-104", "refund window"]
    assert query["language_code"] == ["en"]
    assert scribe.headers == {"xi-api-key": KEY}
    assert KEY not in scribe.url
    assert scribe.closed


async def test_a_custom_api_url_becomes_its_websocket_url():
    scribe = FakeScribe()
    mic = Mic()
    mic.end()

    await collect(stt(Connector(scribe), url="http://localhost:9999").stream(mic), [])

    assert scribe.url.startswith("ws://localhost:9999/v1/speech-to-text/realtime?")


# audio


async def test_frames_are_resampled_to_16k_mono_and_sent_in_tenth_of_a_second_chunks():
    """LiveKit hands over 48 kHz stereo; Scribe gets 16-bit 16 kHz mono."""
    scribe = FakeScribe()
    mic = Mic()
    mic.speak(1.0, rate=48000, channels=2)
    mic.end()

    await collect(stt(Connector(scribe)).stream(mic), [])

    chunks = scribe.audio()
    assert all(m["message_type"] == "input_audio_chunk" for m in scribe.sent)
    assert all(m["sample_rate"] == SCRIBE_SAMPLE_RATE for m in scribe.sent)
    assert all(len(base64.b64decode(m["audio_base_64"])) <= 3200 for m in chunks)
    one_second = SCRIBE_SAMPLE_RATE * 2  # 16-bit samples
    assert abs(scribe.audio_bytes() - one_second) <= one_second * 0.05
    assert not any(m["commit"] for m in chunks)


async def test_16k_mono_frames_pass_through_unchanged():
    scribe = FakeScribe()
    mic = Mic()
    mic.speak(0.5)
    mic.end()

    await collect(stt(Connector(scribe)).stream(mic), [])

    assert scribe.audio_bytes() == 16000  # 0.5 s of 16-bit 16 kHz
    assert base64.b64decode(scribe.audio()[0]["audio_base_64"])[:4] == b"\x01\x00\x01\x00"


async def test_a_quiet_track_keeps_the_session_alive():
    scribe = FakeScribe()
    mic = Mic()
    pieces: list[SpeechPiece] = []
    task = asyncio.create_task(
        collect(stt(Connector(scribe), keepalive_seconds=0.02).stream(mic), pieces)
    )

    await until(lambda: len(scribe.sent) >= 2)
    mic.end()
    await task

    keepalives = [m for m in scribe.sent if not m["audio_base_64"] and not m["commit"]]
    assert keepalives


# transcripts


async def test_partials_and_a_committed_final_become_speech_pieces():
    scribe = FakeScribe()
    mic = Mic()
    pieces: list[SpeechPiece] = []
    task = asyncio.create_task(collect(stt(Connector(scribe)).stream(mic), pieces))
    mic.speak(1.5)
    await until(lambda: scribe.audio_bytes() >= 48000)

    scribe.reply(message_type="session_started", session_id="s-1", config={})
    scribe.reply(message_type="partial_transcript", text="Polaris, what is")
    scribe.reply(message_type="committed_transcript", text="Polaris, what is the refund window?")
    scribe.reply(
        message_type="committed_transcript_with_timestamps",
        text="Polaris, what is the refund window?",
        language_code="en",
        words=words(("Polaris,", 0.2, 0.6), ("window?", 1.1, 1.4)),
    )
    await until(lambda: len(pieces) == 2)
    mic.end()
    await task

    partial, final = pieces
    assert (partial.text, partial.is_final) == ("Polaris, what is", False)
    assert partial.end == pytest.approx(1.5)
    assert final == SpeechPiece("Polaris, what is the refund window?", True, 0.2, 1.4)


async def test_the_next_utterance_starts_where_the_last_final_ended():
    scribe = FakeScribe()
    mic = Mic()
    pieces: list[SpeechPiece] = []
    task = asyncio.create_task(collect(stt(Connector(scribe)).stream(mic), pieces))
    mic.speak(3.0)
    await until(lambda: scribe.audio_bytes() >= 96000)

    scribe.reply(
        message_type="committed_transcript_with_timestamps",
        text="First.",
        words=words(("First.", 0.5, 1.0)),
    )
    scribe.reply(message_type="partial_transcript", text="Second")
    await until(lambda: len(pieces) == 2)
    mic.end()
    await task

    assert pieces[1].start == pytest.approx(1.0)
    assert pieces[1].end == pytest.approx(3.0)


async def test_an_empty_commit_and_repeated_partials_produce_nothing():
    scribe = FakeScribe()
    mic = Mic()
    pieces: list[SpeechPiece] = []
    task = asyncio.create_task(collect(stt(Connector(scribe)).stream(mic), pieces))
    mic.speak(0.2)

    scribe.reply(message_type="partial_transcript", text="")
    scribe.reply(message_type="committed_transcript_with_timestamps", text="  ", words=[])
    scribe.reply(message_type="warning", error="audio is quiet")
    scribe.reply(message_type="partial_transcript", text="Hi")
    scribe.reply(message_type="partial_transcript", text="Hi")
    await until(lambda: len(pieces) == 1)
    await asyncio.sleep(0.05)
    mic.end()
    await task

    assert [p.text for p in pieces] == ["Hi", "Hi"]  # the partial, then finalised at the end
    assert [p.is_final for p in pieces] == [False, True]


# the end of the input


async def test_the_end_of_the_input_commits_and_waits_for_the_last_final():
    scribe = FakeScribe()
    scribe.on_commit = lambda: scribe.reply(
        message_type="committed_transcript_with_timestamps",
        text="That's all from me.",
        words=words(("That's", 0.1, 0.3), ("me.", 0.8, 1.0)),
    )
    mic = Mic()
    mic.speak(1.0)
    mic.end()
    scribe.reply(message_type="partial_transcript", text="That's all")
    pieces: list[SpeechPiece] = []

    await collect(stt(Connector(scribe), finalize_seconds=5).stream(mic), pieces)

    assert pieces[-1] == SpeechPiece("That's all from me.", True, 0.1, 1.0)
    commits = [m for m in scribe.sent if m["commit"]]
    assert len(commits) == 1 and scribe.sent[-1]["commit"]
    assert scribe.closed


async def test_when_scribe_never_commits_the_last_partial_is_kept_as_final():
    scribe = FakeScribe()
    mic = Mic()
    pieces: list[SpeechPiece] = []
    task = asyncio.create_task(collect(stt(Connector(scribe)).stream(mic), pieces))
    mic.speak(0.5)
    scribe.reply(message_type="partial_transcript", text="Okay, that's all")
    await until(lambda: pieces)

    mic.end()
    async with asyncio.timeout(2):
        await task

    assert pieces[-1].text == "Okay, that's all"
    assert pieces[-1].is_final


async def test_the_connection_closing_after_the_input_ended_is_a_clean_end():
    scribe = FakeScribe()
    scribe.on_commit = scribe.drop
    mic = Mic()
    mic.speak(0.2)
    mic.end()

    await collect(stt(Connector(scribe), finalize_seconds=5).stream(mic), [])


# failures


async def test_a_dropped_connection_raises():
    scribe = FakeScribe()
    mic = Mic()
    mic.speak(0.2)
    scribe.drop()

    with pytest.raises(ScribeError):
        await collect(stt(Connector(scribe)).stream(mic), [])
    assert scribe.closed


@pytest.mark.parametrize(
    "message_type", ["auth_error", "quota_exceeded", "rate_limited", "transcriber_error", "error"]
)
async def test_a_scribe_error_raises_without_the_api_key(message_type):
    scribe = FakeScribe()
    mic = Mic()
    mic.speak(0.2)
    scribe.reply(message_type=message_type, error="something went wrong")

    with pytest.raises(ScribeError) as caught:
        await collect(stt(Connector(scribe)).stream(mic), [])

    assert message_type in str(caught.value)
    assert KEY not in str(caught.value)


async def test_failing_to_connect_raises():
    async def refuse(url, headers):
        raise OSError("connection refused")

    mic = Mic()
    mic.end()
    with pytest.raises(OSError):
        await collect(stt(refuse).stream(mic), [])


def test_full_width_punctuation_is_normalised_so_the_wake_phrase_is_found():
    """Scribe has returned full-width commas and question marks for English speech."""
    final = parse_event(
        json.dumps(
            {
                "message_type": "committed_transcript_with_timestamps",
                "text": "Polaris\uff0cwhat is the refund window\uff1f",
                "words": words(("Polaris\uff0c", 0.0, 0.5), ("window\uff1f", 1.0, 1.5)),
            }
        )
    )
    partial = parse_event(
        json.dumps({"message_type": "partial_transcript", "text": "Polaris\uff0c"})
    )

    assert final.text == "Polaris, what is the refund window?"
    assert partial.text == "Polaris,"
    inv = WakeDetector(["Polaris"]).on_segment(
        TranscriptSegment(
            seg_id="s",
            meeting_id="m-1",
            speaker_id="u-alex",
            speaker_name="Alex Chen",
            text=final.text,
            is_final=True,
            t_start=0.0,
            t_end=1.5,
        )
    )
    assert inv is not None and inv.question == "what is the refund window?"


def test_events_are_parsed_by_message_type():
    assert parse_event(json.dumps({"message_type": "partial_transcript", "text": "a"})).kind == (
        "partial"
    )
    final = parse_event(
        json.dumps(
            {
                "message_type": "committed_transcript_with_timestamps",
                "text": "a b",
                "words": words(("a", 1.0, 1.2), ("b", 1.5, 2.0)),
            }
        )
    )
    assert (final.kind, final.text, final.start, final.end) == ("final", "a b", 1.0, 2.0)
    assert parse_event(json.dumps({"message_type": "committed_transcript", "text": "a"})).kind == (
        "ignore"
    )
    assert parse_event(json.dumps({"message_type": "auth_error", "error": "bad"})).kind == "error"
    assert parse_event("not json").kind == "ignore"


# reconnecting through the manager


class Bus:
    def __init__(self):
        self.published = []

    async def publish(self, topic, payload, *, to=None):
        self.published.append(payload)


class Brain:
    def __init__(self):
        self.saved = []

    async def ingest_segments(self, meeting_id, segments):
        self.saved += segments


async def test_the_manager_reopens_a_dropped_scribe_session_on_the_same_track():
    first, second = FakeScribe(), FakeScribe()
    connector = Connector(first, second)
    bus, brain = Bus(), Brain()
    invocations: list[Invocation] = []

    async def on_invocation(invocation):
        invocations.append(invocation)

    manager = TranscriptionManager(
        "m-1",
        stt=stt(connector),
        bus=bus,
        brain=brain,
        detector=WakeDetector(["Polaris"]),
        on_invocation=on_invocation,
        clock=lambda: 10.0,
        restart_backoff=0,
        drain_seconds=1.0,
    )
    mic = Mic()
    mic.speak(0.3)
    manager.start("u-alex", "Alex Chen", "TR_alex", mic)
    await until(lambda: first.audio())
    first.drop()
    await until(lambda: len(connector.opened) == 2)
    mic.speak(0.3)
    second.reply(
        message_type="committed_transcript_with_timestamps",
        text="Polaris, what is the refund window?",
        words=words(("Polaris,", 0.0, 0.4), ("window?", 1.0, 1.5)),
    )
    await until(lambda: brain.saved)
    await manager.stop("TR_alex")
    await manager.aclose()

    [saved] = brain.saved
    assert (saved.speaker_id, saved.text, saved.t_start) == (
        "u-alex",
        "Polaris, what is the refund window?",
        10.0,
    )
    assert [i.question for i in invocations] == ["what is the refund window?"]
