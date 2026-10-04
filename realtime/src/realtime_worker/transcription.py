"""One speech-to-text session per participant audio track.

The manager owns what the rest of the system relies on: the LiveKit identity of each segment,
a seg_id unique across sessions (one per utterance, shared by its partials and its final), and
times in seconds from the meeting start. Every caption goes to the room on Topic.TRANSCRIPT;
final segments are saved through the brain and checked for a deliberate invocation, in the
background so a slow brain never delays captions.

Words are not lost quietly: a dropped transcriber connection is reopened, saves that fail are
held and resent (the brain ignores an identical seg_id), stopping a track lets the last
sentence finish, and whatever could not be saved is counted.
"""

import asyncio
import logging
import time
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable, Coroutine
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from contracts import AGENT_PARTICIPANT_ID, Invocation, Topic, TranscriptSegment

from .brain_client import BrainClient, BrainRejected
from .invocation import WakeDetector
from .room import RoomBus
from .stt import SpeechPiece, SpeechToText

log = logging.getLogger(__name__)

# Final segments kept as context for an invocation; the brain takes at most this many.
RECENT_FINALS = 20


@dataclass
class Session:
    participant_id: str
    task: asyncio.Task[None]
    stopped: asyncio.Event = field(default_factory=asyncio.Event)


class TranscriptionManager:
    def __init__(
        self,
        meeting_id: str,
        *,
        stt: SpeechToText,
        bus: RoomBus,
        brain: BrainClient,
        detector: WakeDetector,
        on_invocation: Callable[[Invocation], Awaitable[None]],
        on_unavailable: Callable[[str, str], Awaitable[None]] | None = None,
        clock: Callable[[], float] | None = None,
        max_restarts: int = 3,
        restart_backoff: float = 1.0,
        drain_seconds: float = 2.0,
        backlog_limit: int = 1000,
    ):
        """clock() returns seconds since the meeting started (Meeting.started_at); every
        segment time and the Ask window are on it. on_unavailable(participant_id, name) is
        called when a participant's captions stop for good."""
        self.meeting_id = meeting_id
        self._stt = stt
        self._bus = bus
        self._brain = brain
        self._detector = detector
        self._on_invocation = on_invocation
        self._on_unavailable = on_unavailable
        if clock is None:
            origin = time.monotonic()
            clock = lambda: time.monotonic() - origin  # noqa: E731
        self._clock = clock
        self._max_restarts = max_restarts
        self._restart_backoff = restart_backoff
        self._drain_seconds = drain_seconds
        self._backlog_limit = backlog_limit

        self._sessions: dict[str, Session] = {}
        self._background: set[asyncio.Task[None]] = set()
        self._save_lock = asyncio.Lock()
        self._backlog: list[TranscriptSegment] = []
        self._saved = 0
        self._dropped = 0
        self._recent: deque[TranscriptSegment] = deque(maxlen=RECENT_FINALS)

    # state

    def active_tracks(self) -> set[str]:
        return set(self._sessions)

    def saved_count(self) -> int:
        return self._saved

    def recent_finals(self) -> list[TranscriptSegment]:
        """The latest final segments of everyone, oldest first: an invocation's context."""
        return list(self._recent)

    def unsaved_count(self) -> int:
        """Final segments not in the brain's record: still held, or given up on. Nonzero means
        the transcript is incomplete and the report should say so."""
        return len(self._backlog) + self._dropped

    # control

    def start(self, participant_id: str, participant_name: str, track_sid: str, audio) -> None:
        """Start transcribing a microphone track. The agent's own audio is never transcribed."""
        if participant_id == AGENT_PARTICIPANT_ID or track_sid in self._sessions:
            return
        stopped = asyncio.Event()
        task = asyncio.create_task(
            self._run(participant_id, participant_name, audio, stopped),
            name=f"stt:{participant_id}:{track_sid}",
        )
        session = Session(participant_id, task, stopped)
        self._sessions[track_sid] = session
        task.add_done_callback(lambda _: self._forget(track_sid, session))

    def arm_ask(self, participant_id: str) -> None:
        """The Ask button: this participant's next final segment is the question."""
        self._detector.arm_ask(participant_id, at=self._clock())

    def cancel_ask(self, participant_id: str) -> bool:
        """Withdraw an Ask press. True if one was waiting."""
        return self._detector.cancel_ask(participant_id)

    async def stop(self, track_sid: str) -> None:
        """The track was muted, unpublished or unsubscribed. Ends its audio so the transcriber
        can finalise the last sentence, waiting up to drain_seconds before cancelling."""
        session = self._sessions.get(track_sid)
        if session is None:
            return
        session.stopped.set()
        try:
            await asyncio.wait_for(asyncio.shield(session.task), self._drain_seconds)
        except TimeoutError:
            await cancel(session.task)
        self._forget(track_sid, session)

    async def stop_participant(self, participant_id: str) -> None:
        tracks = [t for t, s in self._sessions.items() if s.participant_id == participant_id]
        await asyncio.gather(*(self.stop(t) for t in tracks))

    async def join(self) -> None:
        """Wait until every session has ended and every save and invocation has finished,
        then try once more to save anything still held."""
        while pending := [
            t
            for t in [*(s.task for s in self._sessions.values()), *self._background]
            if not t.done()
        ]:
            await asyncio.wait(pending)
        await self._flush()

    async def aclose(self) -> None:
        """End every session letting its last sentence finish, wait briefly for in-flight saves,
        flush what is held, then cancel what remains."""
        await asyncio.gather(*(self.stop(t) for t in list(self._sessions)))
        if pending := [t for t in self._background if not t.done()]:
            await asyncio.wait(pending, timeout=self._drain_seconds)
        await self._flush()
        for task in list(self._background):
            await cancel(task)

    # sessions

    async def _run(self, participant_id: str, name: str, audio, stopped: asyncio.Event) -> None:
        failures = 0
        while not stopped.is_set():
            origin = self._clock()
            utterance: str | None = None
            try:
                async for piece in self._stt.stream(until_set(audio, stopped)):
                    failures = 0
                    utterance = utterance or f"{self.meeting_id}-{uuid4().hex}"
                    segment = self._segment(piece, utterance, participant_id, name, origin)
                    if piece.is_final:
                        utterance = None
                    await self._handle(segment)
                return
            except asyncio.CancelledError:
                raise
            except Exception:
                failures += 1
                log.warning(
                    "Transcriber dropped for %s (%d in a row)",
                    participant_id,
                    failures,
                    exc_info=True,
                )
                if failures > self._max_restarts:
                    await self._unavailable(participant_id, name)
                    return
                await asyncio.sleep(self._restart_backoff * 2 ** (failures - 1))

    def _segment(
        self, piece: SpeechPiece, seg_id: str, participant_id: str, name: str, origin: float
    ) -> TranscriptSegment:
        return TranscriptSegment(
            seg_id=seg_id,
            meeting_id=self.meeting_id,
            speaker_id=participant_id,
            speaker_name=name,
            text=piece.text,
            is_final=piece.is_final,
            t_start=origin + piece.start,
            t_end=origin + piece.end,
        )

    async def _handle(self, segment: TranscriptSegment) -> None:
        try:
            await self._bus.publish(Topic.TRANSCRIPT, segment)
        except Exception:
            log.exception("Could not publish caption %s", segment.seg_id)
        if segment.is_final:
            self._recent.append(segment)
            self._spawn(self._save(segment))
            if invocation := self._detector.on_segment(segment):
                self._spawn(self._invoke(invocation))

    async def _unavailable(self, participant_id: str, name: str) -> None:
        log.error("Captions unavailable for %s after repeated transcriber drops", participant_id)
        if self._on_unavailable:
            try:
                await self._on_unavailable(participant_id, name)
            except Exception:
                log.exception("Could not report captions unavailable for %s", participant_id)

    # saving

    async def _save(self, segment: TranscriptSegment) -> None:
        async with self._save_lock:
            await self._send([*self._backlog, segment])

    async def _flush(self) -> None:
        async with self._save_lock:
            if self._backlog:
                await self._send(list(self._backlog))

    async def _send(self, batch: list[TranscriptSegment]) -> None:
        """Save the batch; on a rejection, save one by one so a bad segment can't block the
        rest. What fails for now is held; what is rejected is counted as dropped."""
        try:
            await self._brain.ingest_segments(self.meeting_id, batch)
            self._saved += len(batch)
            self._backlog = []
            return
        except BrainRejected:
            held = []
            for segment in batch:
                try:
                    await self._brain.ingest_segments(self.meeting_id, [segment])
                    self._saved += 1
                except BrainRejected as e:
                    self._dropped += 1
                    log.warning("Brain rejected segment %s (%d)", segment.seg_id, e.status)
                except Exception:
                    held.append(segment)
            self._hold(held)
        except Exception as e:
            log.warning("Holding %d segment(s) to resend: %s", len(batch), e)
            self._hold(batch)

    def _hold(self, segments: list[TranscriptSegment]) -> None:
        overflow = max(0, len(segments) - self._backlog_limit)
        if overflow:
            log.error("Backlog full; dropping %d oldest segment(s)", overflow)
        self._dropped += overflow
        self._backlog = segments[overflow:]

    # background work

    async def _invoke(self, invocation: Invocation) -> None:
        try:
            await self._on_invocation(invocation)
        except Exception:
            log.exception("Invocation %s failed", invocation.id)

    def _spawn(self, work: Coroutine[Any, Any, None]) -> None:
        task = asyncio.create_task(work)
        self._background.add(task)
        task.add_done_callback(self._background.discard)

    def _forget(self, track_sid: str, session: Session) -> None:
        if self._sessions.get(track_sid) is session:
            del self._sessions[track_sid]


async def until_set(audio, stopped: asyncio.Event) -> AsyncIterator[Any]:
    """The track's frames until `stopped` is set; then the input ends, so the transcriber can
    finalise. A fresh wrapper per stream lets a reopened stream read the same track."""
    if stopped.is_set():
        return
    stop = asyncio.ensure_future(stopped.wait())
    try:
        while True:
            frame = asyncio.ensure_future(anext(audio))
            done, _ = await asyncio.wait({frame, stop}, return_when=asyncio.FIRST_COMPLETED)
            if frame not in done:
                frame.cancel()
                return
            try:
                yield frame.result()
            except StopAsyncIteration:
                return
    finally:
        stop.cancel()


async def cancel(task: asyncio.Task[None]) -> None:
    if task.done():
        return
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        if not task.cancelled():
            raise
