"""One speech-to-text session per participant audio track.

The manager owns what the rest of the system relies on: the LiveKit identity of each segment,
a seg_id unique across sessions (one per utterance, shared by its partials and its final), and
times in seconds from the meeting start. Every caption goes to the room on Topic.TRANSCRIPT;
final segments are saved through the brain and checked for a deliberate invocation, in the
background so a slow brain never delays captions.

Words are not lost quietly: a dropped transcriber connection is reopened, saves that fail are
held and resent (the brain ignores an identical seg_id), stopping a track lets the last
sentence finish, and whatever could not be saved is counted.

A spoken question that trails off is held by the detector for the rest of the sentence; the
manager owns the clock, so it runs the timer that sends a held question when nothing follows.

With a translator (#106) the room reads only English. A finished sentence in another language is
translated before it is published, saved or checked for the wake word. A sentence still going
after provisional_seconds is translated so far and published as a partial, again each period
while it grows; its final replaces it in place (same seg_id). Once a speaker is known to speak
another language their original partials are hidden. A provisional answer arriving after a newer
one, or after the sentence finished, is dropped. If translation fails, the original is shown and
saved with its language and no original_text: untranslated, never invented.
"""

import asyncio
import logging
import math
import time
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable, Coroutine
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from contracts import (
    AGENT_PARTICIPANT_ID,
    Invocation,
    Topic,
    TranscriptSegment,
    TranslateResponse,
)
from contracts.language import normalise_language

from .brain_client import BrainClient, BrainRejected
from .invocation import WakeDetector
from .room import RoomBus
from .stt import SpeechPiece, SpeechToText

log = logging.getLogger(__name__)

# Final segments kept as context for an invocation; the brain takes at most this many.
RECENT_FINALS = 20

# translate(text, language) -> the language and the English; language is a hint or None.
Translate = Callable[[str, str | None], Awaitable[TranslateResponse]]
# A final this short ("Okay.", "Sí") never changes what language a speaker is known to speak.
LANGUAGE_MIN_WORDS = 3


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
        translate: Translate | None = None,
        provisional_seconds: float = 1.5,
        translation_pause_seconds: float = 45.0,
    ):
        """clock() returns seconds since the meeting started (Meeting.started_at); every
        segment time and the Ask window are on it. on_unavailable(participant_id, name) is
        called when a participant's captions stop for good. translate turns non-English speech
        into English (#106); without it every utterance is shown and saved as heard."""
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
        self._translate = translate
        self._provisional_seconds = provisional_seconds
        self._languages: dict[str, str] = {}  # participant -> the language they speak, once known
        self._translation_pause_seconds = translation_pause_seconds
        self._translation_off = False  # no model configured: stop asking for this meeting
        self._provisional_paused_until = 0.0  # time.monotonic(); after a failed translation

        self._sessions: dict[str, Session] = {}
        self._background: set[asyncio.Task[None]] = set()
        self._save_lock = asyncio.Lock()
        self._backlog: list[TranscriptSegment] = []
        self._saved = 0
        self._dropped = 0
        self._recent: deque[TranscriptSegment] = deque(maxlen=RECENT_FINALS)
        self._held_timer: asyncio.Task[None] | None = None
        self._held_deadline: float | None = None

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
            for t in [
                *(s.task for s in self._sessions.values()),
                *self._background,
                *([self._held_timer] if self._held_timer else []),
            ]
            if not t.done()
        ]:
            await asyncio.wait(pending)
        await self._flush()

    async def aclose(self) -> None:
        """End every session letting its last sentence finish, wait briefly for in-flight saves,
        flush what is held, then cancel what remains."""
        await asyncio.gather(*(self.stop(t) for t in list(self._sessions)))
        if self._held_timer:
            self._held_timer.cancel()
        self._send_held(math.inf)  # a question still waiting for its last words goes as it is
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
            live: OpenSentence | None = None
            try:
                async for piece in self._stt.stream(until_set(audio, stopped)):
                    failures = 0
                    utterance = utterance or f"{self.meeting_id}-{uuid4().hex}"
                    segment = self._segment(piece, utterance, participant_id, name, origin)
                    if piece.is_final:
                        utterance = None
                        if live:
                            live.close()
                            live = None
                        await self._handle(await self._in_english(segment, piece.language))
                    elif self._translate is None:
                        await self._handle(segment)
                    else:
                        if live is None or live.seg_id != segment.seg_id:
                            live = OpenSentence(self, segment)
                        live.latest = segment
                        if self._show_original(participant_id, live):
                            await self._handle(segment)
                return
            except asyncio.CancelledError:
                raise
            except Exception:
                if live:
                    live.close()  # not after the backoff below: the dropped stream is done
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
            finally:
                if live:  # however the stream ended, its open sentence stops translating
                    live.close()

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

    async def _publish(self, segment: TranscriptSegment) -> None:
        try:
            await self._bus.publish(Topic.TRANSCRIPT, segment)
        except Exception:
            log.exception("Could not publish caption %s", segment.seg_id)

    async def _handle(self, segment: TranscriptSegment) -> None:
        await self._publish(segment)
        if segment.is_final:
            self._recent.append(segment)
            self._spawn(self._save(segment))
            if invocation := self._detector.on_segment(segment):
                self._spawn(self._invoke(invocation))
            self._time_held()

    async def _unavailable(self, participant_id: str, name: str) -> None:
        log.error("Captions unavailable for %s after repeated transcriber drops", participant_id)
        if self._on_unavailable:
            try:
                await self._on_unavailable(participant_id, name)
            except Exception:
                log.exception("Could not report captions unavailable for %s", participant_id)

    # translation (#106)

    def _speaks_another_language(self, participant_id: str) -> bool:
        language = self._languages.get(participant_id)
        return language is not None and language != "en"

    def _translating(self) -> bool:
        return self._translate is not None and not self._translation_off

    def _provisional_paused(self) -> bool:
        return time.monotonic() < self._provisional_paused_until

    def _show_original(self, participant_id: str, live: "OpenSentence") -> bool:
        """The speaker's own words, unless a translation is on its way: they speak another
        language and translation is working. Never a blank caption while it isn't."""
        if not self._speaks_another_language(participant_id) or not self._translating():
            return True
        return live.failing or live.english or self._provisional_paused()

    def _translation_failed(self, error: Exception) -> None:
        """Back off: no model configured (503) ends translation for the meeting; any other
        failure (502, 504, a timeout) pauses provisional translation for a while."""
        if getattr(error, "status", None) == 503:
            if not self._translation_off:
                log.error("Live translation is off for meeting %s: %s", self.meeting_id, error)
            self._translation_off = True
        else:
            self._provisional_paused_until = time.monotonic() + self._translation_pause_seconds

    def _learn_language(self, participant_id: str, language: str | None, text: str) -> None:
        """A final of LANGUAGE_MIN_WORDS or more sets the language a speaker is known to speak;
        a provisional answer only when nothing is known yet."""
        if language and len(text.split()) >= LANGUAGE_MIN_WORDS:
            self._languages[participant_id] = language

    async def _in_english(
        self, segment: TranscriptSegment, detected: str | None
    ) -> TranscriptSegment:
        """A finished sentence as the room should read it. English (or no translator, or no
        language known) is unchanged; otherwise translated, or kept untranslated (its language
        set, no original_text) when translation is off or fails."""
        language = normalise_language(detected) or self._languages.get(segment.speaker_id)
        self._learn_language(segment.speaker_id, language, segment.text)
        if self._translate is None or language in (None, "en"):
            return segment
        english = ""
        if not self._translation_off:
            try:
                english = (await self._translate(segment.text, language)).text.strip()
            except Exception as e:
                log.warning("Could not translate %s (%s): %s", segment.seg_id, language, e)
                self._translation_failed(e)
        if not english:
            return segment.model_copy(update={"language": language})
        return segment.model_copy(
            update={"text": english, "original_text": segment.text, "language": language}
        )

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

    # held questions

    def _time_held(self) -> None:
        """Keep one timer running for the earliest question the detector is holding."""
        deadline = self._detector.next_due()
        if deadline == self._held_deadline and self._held_timer and not self._held_timer.done():
            return
        if self._held_timer and not self._held_timer.done():
            self._held_timer.cancel()
        self._held_timer, self._held_deadline = None, deadline
        if deadline is not None:
            self._held_timer = asyncio.create_task(self._send_held_at(deadline))

    async def _send_held_at(self, deadline: float) -> None:
        await asyncio.sleep(max(0.0, deadline - self._clock()))
        self._held_timer = None
        self._send_held(max(self._clock(), deadline))
        self._time_held()

    def _send_held(self, now: float) -> None:
        for invocation in self._detector.due(now):
            self._spawn(self._invoke(invocation))

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


class OpenSentence:
    """A sentence still being spoken, translated provisionally every provisional_seconds while
    it grows. At most one request in flight; an answer is shown only if it is the newest and the
    sentence is still open."""

    def __init__(self, manager: TranscriptionManager, first: TranscriptSegment):
        self.manager = manager
        self.seg_id = first.seg_id
        self.latest = first
        self.requested: str | None = None
        self.sent = 0
        self.shown = 0
        self.in_flight = False
        self.failing = False  # the latest request failed: the speaker's own words are shown
        self.english = False  # the model found this sentence English: shown as said
        self.closed = False
        self.timer = asyncio.create_task(self._tick())

    def close(self) -> None:
        self.closed = True
        self.timer.cancel()

    async def _tick(self) -> None:
        m = self.manager
        while not self.closed:
            await asyncio.sleep(m._provisional_seconds)
            if m._languages.get(self.latest.speaker_id) == "en" or m._translation_off:
                return  # the original is what the room reads
            if m._provisional_paused():
                continue
            if not self.in_flight and self.latest.text != self.requested:
                m._spawn(self._request(self.latest))

    async def _request(self, segment: TranscriptSegment) -> None:
        m = self.manager
        self.in_flight, self.requested = True, segment.text
        self.sent += 1
        number = self.sent
        try:
            # No hint: the speaker may have switched language; the model detects it.
            answer = await m._translate(segment.text, None)  # type: ignore[misc]
        except Exception as e:
            log.info("Provisional translation of %s failed: %s", segment.seg_id, e)
            m._translation_failed(e)
            if not self.closed and not self.failing:
                self.failing = True
                await m._publish(self.latest)  # the speaker's own words, not a blank caption
            return
        finally:
            self.in_flight = False
        if self.closed or number <= self.shown:
            return  # the sentence finished, or a newer answer was shown
        self.shown, self.failing = number, False
        language = normalise_language(answer.language)
        if language is None:
            return
        was_hidden = not m._show_original(segment.speaker_id, self)
        m._languages.setdefault(segment.speaker_id, language)  # finals decide once known
        if language == "en":
            self.english = True
            if was_hidden:  # switched to English mid-sentence: show it as said
                await m._publish(self.latest)
            return
        if not answer.text.strip():
            return
        await m._publish(
            segment.model_copy(
                update={
                    "text": answer.text.strip(),
                    "original_text": segment.text,
                    "language": language,
                }
            )
        )


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
