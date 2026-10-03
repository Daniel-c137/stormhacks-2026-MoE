from collections.abc import AsyncIterator
from typing import Protocol

from livekit import rtc

from contracts import TranscriptSegment


class SpeechToText(Protocol):
    """ElevenLabs Scribe v2 Realtime, one stream per participant audio track.

    Yields partial segments for captions, then one final segment per utterance.
    """

    def stream(
        self,
        meeting_id: str,
        speaker_id: str,
        speaker_name: str,
        audio: AsyncIterator[rtc.AudioFrame],
    ) -> AsyncIterator[TranscriptSegment]: ...
