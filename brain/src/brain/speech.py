"""The agent's voice through ElevenLabs text-to-speech, for the report page's Listen button.
Never plays into a live meeting."""

import asyncio
import hashlib
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from urllib.parse import quote

import httpx

from .config import Settings

OUTPUT_FORMAT = "mp3_44100_128"
CONTENT_TYPE = "audio/mpeg"
# A long summary takes a while to speak; connecting should not.
TIMEOUT = httpx.Timeout(90.0, connect=5.0)


class SpeechUnavailable(RuntimeError):
    """ElevenLabs speech is not configured."""


class SpeechFailed(RuntimeError):
    """ElevenLabs is configured but the call failed. The message never holds the text."""


def missing(config: Settings, voice_id: str | None) -> list[str]:
    """The settings speech still needs, by their environment names."""
    needed = {
        "ELEVENLABS_API_KEY": config.elevenlabs_api_key,
        "ELEVENLABS_TTS_MODEL": config.elevenlabs_tts_model,
        "ELEVENLABS_VOICE_ID": voice_id,
    }
    return [name for name, value in needed.items() if not value]


def audio_key(text: str, voice_id: str, model: str) -> str:
    """Names audio by what it was made from, so a changed text, voice or model makes it again."""
    return hashlib.sha256(json.dumps([text, voice_id, model]).encode()).hexdigest()


async def synthesize(
    config: Settings,
    voice_id: str | None,
    text: str,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
) -> bytes:
    """MP3 of `text` in the voice, exactly as given (POST /v1/text-to-speech/{voice_id})."""
    if gaps := missing(config, voice_id):
        raise SpeechUnavailable(f"ElevenLabs speech is not configured: set {', '.join(gaps)}")
    async with httpx.AsyncClient(
        base_url=config.elevenlabs_api_url,
        headers={"xi-api-key": config.elevenlabs_api_key or ""},
        timeout=TIMEOUT,
        transport=transport,
    ) as client:
        try:
            response = await client.post(
                # The voice comes from team settings: one escaped path segment, never a path.
                f"/v1/text-to-speech/{quote(voice_id or '', safe='')}",
                params={"output_format": OUTPUT_FORMAT},
                json={"text": text, "model_id": config.elevenlabs_tts_model},
                headers={"accept": CONTENT_TYPE},
            )
        except httpx.HTTPError as e:
            raise SpeechFailed(f"Could not reach ElevenLabs: {str(e) or type(e).__name__}") from e
    if response.is_error:  # the error body may quote the text, so only the status is kept
        raise SpeechFailed(f"ElevenLabs refused the speech request ({response.status_code})")
    if not response.content:
        raise SpeechFailed("ElevenLabs returned no audio")
    return response.content


class MeetingLocks:
    """One lock per meeting, so a process makes a meeting's audio at most once at a time. A lock
    is dropped once nobody holds or waits for it."""

    def __init__(self):
        self._locks: dict[str, tuple[asyncio.Lock, int]] = {}

    @asynccontextmanager
    async def hold(self, meeting_id: str) -> AsyncIterator[None]:
        lock, users = self._locks.get(meeting_id, (asyncio.Lock(), 0))
        self._locks[meeting_id] = (lock, users + 1)
        try:
            async with lock:
                yield
        finally:
            lock, users = self._locks[meeting_id]
            if users == 1:
                del self._locks[meeting_id]
            else:
                self._locks[meeting_id] = (lock, users - 1)
