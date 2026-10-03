from collections.abc import AsyncIterator
from typing import Protocol

from livekit import rtc


class TextToSpeech(Protocol):
    """ElevenLabs TTS. Only ever called after a participant chooses Speak."""

    def synthesize(self, text: str, voice_id: str) -> AsyncIterator[rtc.AudioFrame]: ...
