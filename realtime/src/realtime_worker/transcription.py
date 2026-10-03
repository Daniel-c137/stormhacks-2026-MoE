"""One speech-to-text session per participant audio track.

Every caption goes to the room on Topic.TRANSCRIPT; final segments are saved through the brain
and checked for a deliberate invocation. Saving and invocations run in the background so a
slow brain never delays captions. A failing session, save or invocation is logged and contained.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import dataclass
from typing import Any

from contracts import AGENT_PARTICIPANT_ID, Invocation, Topic, TranscriptSegment

from .brain_client import BrainClient
from .invocation import WakeDetector
from .room import RoomBus
from .stt import SpeechToText

log = logging.getLogger(__name__)


@dataclass
class Session:
    participant_id: str
    task: asyncio.Task[None]


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
    ):
        self.meeting_id = meeting_id
        self._stt = stt
        self._bus = bus
        self._brain = brain
        self._detector = detector
        self._on_invocation = on_invocation
        self._sessions: dict[str, Session] = {}
        self._background: set[asyncio.Task[None]] = set()

    def active_tracks(self) -> set[str]:
        return set(self._sessions)

    def start(self, participant_id: str, participant_name: str, track_sid: str, audio) -> None:
        """Start transcribing a microphone track. The agent's own audio is never transcribed."""
        if participant_id == AGENT_PARTICIPANT_ID or track_sid in self._sessions:
            return
        task = asyncio.create_task(
            self._run(participant_id, participant_name, track_sid, audio),
            name=f"stt:{participant_id}:{track_sid}",
        )
        session = Session(participant_id, task)
        self._sessions[track_sid] = session
        task.add_done_callback(lambda _: self._forget(track_sid, session))

    async def stop(self, track_sid: str) -> None:
        """The track was muted, unpublished or unsubscribed."""
        session = self._sessions.pop(track_sid, None)
        if session:
            await cancel(session.task)

    async def stop_participant(self, participant_id: str) -> None:
        for track_sid, session in list(self._sessions.items()):
            if session.participant_id == participant_id:
                await self.stop(track_sid)

    async def join(self) -> None:
        """Wait until every session has ended and every save and invocation has finished."""
        while pending := [
            t
            for t in [*(s.task for s in self._sessions.values()), *self._background]
            if not t.done()
        ]:
            await asyncio.wait(pending)

    async def aclose(self) -> None:
        sessions, self._sessions = list(self._sessions.values()), {}
        for task in [*(s.task for s in sessions), *self._background]:
            await cancel(task)

    async def _run(self, participant_id: str, name: str, track_sid: str, audio) -> None:
        try:
            async for segment in self._stt.stream(self.meeting_id, participant_id, name, audio):
                await self._caption(segment)
                if segment.is_final:
                    self._spawn(self._save(segment))
                    if invocation := self._detector.on_segment(segment):
                        self._spawn(self._invoke(invocation))
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("Transcription stopped for %s (%s)", participant_id, track_sid)

    async def _caption(self, segment: TranscriptSegment) -> None:
        try:
            await self._bus.publish(Topic.TRANSCRIPT, segment)
        except Exception:
            log.exception("Could not publish caption %s", segment.seg_id)

    async def _save(self, segment: TranscriptSegment) -> None:
        try:
            await self._brain.ingest_segments(self.meeting_id, [segment])
        except Exception:
            log.exception("Could not save segment %s", segment.seg_id)

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


async def cancel(task: asyncio.Task[None]) -> None:
    if task.done():
        return
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        if not task.cancelled():
            raise
