"""Which LiveKit tracks are transcribed, and when.

Each participant's subscribed microphone is transcribed while it is unmuted: muting, unpublishing
or unsubscribing stops it (the last sentence still finishes), unmuting starts it again on fresh
audio, and leaving stops everything of that participant. Coming back is a new subscription, so it
starts again. Screen-share audio and the agent's own voice are never transcribed.
"""

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from livekit import rtc

from contracts import AGENT_PARTICIPANT_ID

log = logging.getLogger(__name__)


class Audio(Protocol):
    """A track's frames as an async iterator, closed when transcription stops."""

    def __aiter__(self): ...

    async def __anext__(self) -> rtc.AudioFrame: ...

    async def aclose(self) -> None: ...


class Transcription(Protocol):
    def start(self, participant_id: str, participant_name: str, track_sid: str, audio) -> None: ...

    async def stop(self, track_sid: str) -> None: ...

    async def stop_participant(self, participant_id: str) -> None: ...


class TrackAudio:
    """A remote track's audio at 16 kHz mono (what Scribe takes), frame by frame. Reading it is
    cancel-safe, so a transcriber reopened on the same track loses no more than a frame."""

    def __init__(self, track: rtc.Track):
        self._stream = rtc.AudioStream.from_track(track=track, sample_rate=16000, num_channels=1)

    def __aiter__(self):
        return self

    async def __anext__(self) -> rtc.AudioFrame:
        return (await self._stream.__anext__()).frame

    async def aclose(self) -> None:
        await self._stream.aclose()


@dataclass
class Mic:
    participant_id: str
    name: str
    track: Any
    audio: Audio | None = None  # set while transcribing


class TrackRouter:
    def __init__(
        self,
        transcription: Transcription,
        open_audio: Callable[[Any], Audio] = TrackAudio,
    ):
        self._transcription = transcription
        self._open_audio = open_audio
        self._mics: dict[str, Mic] = {}  # track sid -> subscribed microphone
        self._tasks: set[asyncio.Task[None]] = set()

    def attach(self, room: rtc.Room) -> None:
        room.on("track_subscribed", self.subscribed)
        room.on("track_muted", lambda p, pub: self._spawn(self.muted(p, pub)))
        room.on("track_unmuted", self.unmuted)
        room.on("track_unsubscribed", lambda t, pub, p: self._spawn(self.unsubscribed(t, pub, p)))
        room.on("track_unpublished", lambda pub, p: self._spawn(self.unsubscribed(None, pub, p)))
        room.on("participant_disconnected", lambda p: self._spawn(self.left(p)))

    def subscribed(self, track, publication, participant) -> None:
        if participant.identity == AGENT_PARTICIPANT_ID:
            return
        if track.kind != rtc.TrackKind.KIND_AUDIO:
            return
        if publication.source != rtc.TrackSource.SOURCE_MICROPHONE:
            return
        name = participant.name or participant.identity
        self._mics[publication.sid] = Mic(participant.identity, name, track)
        if not publication.muted:
            self._start(publication.sid)

    async def muted(self, participant, publication) -> None:
        await self._stop(publication.sid)

    def unmuted(self, participant, publication) -> None:
        mic = self._mics.get(publication.sid)
        if mic is not None and mic.audio is None:
            self._start(publication.sid)

    async def unsubscribed(self, track, publication, participant) -> None:
        await self._stop(publication.sid)
        self._mics.pop(publication.sid, None)

    async def left(self, participant) -> None:
        await self._transcription.stop_participant(participant.identity)
        for sid, mic in list(self._mics.items()):
            if mic.participant_id == participant.identity:
                await self._close(mic)
                del self._mics[sid]

    async def aclose(self) -> None:
        for mic in self._mics.values():
            await self._close(mic)
        self._mics.clear()
        for task in list(self._tasks):
            await task

    def _start(self, sid: str) -> None:
        mic = self._mics[sid]
        mic.audio = self._open_audio(mic.track)
        self._transcription.start(mic.participant_id, mic.name, sid, mic.audio)
        log.info("Transcribing %s (%s)", mic.participant_id, sid)

    async def _stop(self, sid: str) -> None:
        mic = self._mics.get(sid)
        if mic is None or mic.audio is None:
            return
        audio, mic.audio = mic.audio, None  # an unpublish also unsubscribes: stop once
        await self._transcription.stop(sid)
        await close(audio)
        log.info("Stopped transcribing %s (%s)", mic.participant_id, sid)

    async def _close(self, mic: Mic) -> None:
        if mic.audio is not None:
            audio, mic.audio = mic.audio, None
            await close(audio)

    def _spawn(self, work) -> None:
        task = asyncio.get_running_loop().create_task(work)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)


async def close(audio: Audio) -> None:
    try:
        await audio.aclose()
    except Exception:
        log.debug("Closing an audio stream failed", exc_info=True)
