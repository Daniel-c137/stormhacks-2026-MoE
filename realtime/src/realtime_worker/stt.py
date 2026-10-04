from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Protocol

from livekit import rtc


@dataclass(frozen=True)
class SpeechPiece:
    """Text heard on one stream. start and end are seconds from that stream's start; the
    TranscriptionManager moves them onto the meeting clock and attaches identity and seg_id."""

    text: str
    is_final: bool
    start: float
    end: float


class SpeechToText(Protocol):
    """ElevenLabs Scribe v2 Realtime: one stream per participant audio track.

    Yields partial pieces for captions, then one final piece per utterance. When the audio
    input ends, it finalises the utterance in progress and then ends the stream. Raises if the
    connection drops; the manager reopens it.
    """

    def stream(self, audio: AsyncIterator[rtc.AudioFrame]) -> AsyncIterator[SpeechPiece]: ...
