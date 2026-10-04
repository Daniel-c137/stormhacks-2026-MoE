"""ElevenLabs text-to-speech as LiveKit audio frames, for the agent's published track. Called only
after a participant chooses Speak on an answer card."""

import re
from collections.abc import AsyncIterator
from urllib.parse import quote

import httpx
from livekit import rtc

TTS_SAMPLE_RATE = 24000
FRAME_SAMPLES = TTS_SAMPLE_RATE // 50  # 20 ms
FRAME_BYTES = FRAME_SAMPLES * 2  # 16-bit mono
TIMEOUT = httpx.Timeout(30.0, connect=5.0)


class SpeechUnavailable(RuntimeError):
    """Speech is not configured (no voice)."""


class SpeechFailed(RuntimeError):
    """ElevenLabs is configured but the call failed. The message never holds the text."""


class ElevenLabsTTS:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        url: str = "https://api.elevenlabs.io",
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self._api_key = api_key
        self._model = model
        self._url = url
        self._transport = transport

    async def synthesize(self, text: str, voice_id: str) -> AsyncIterator[rtc.AudioFrame]:
        """Raw 24 kHz PCM streamed from POST /v1/text-to-speech/{voice}/stream, as 20 ms frames
        (the last one padded with silence)."""
        if not voice_id:
            raise SpeechUnavailable("No voice: set ELEVENLABS_VOICE_ID or choose a team voice")
        buffer = b""
        heard = False
        async with httpx.AsyncClient(
            base_url=self._url,
            headers={"xi-api-key": self._api_key},
            timeout=TIMEOUT,
            transport=self._transport,
        ) as client:
            try:
                async with client.stream(
                    "POST",
                    # The voice may come from team settings: one escaped path segment.
                    f"/v1/text-to-speech/{quote(voice_id, safe='')}/stream",
                    params={"output_format": f"pcm_{TTS_SAMPLE_RATE}"},
                    json={"text": text, "model_id": self._model},
                ) as response:
                    if response.is_error:  # the error body may quote the text: status only
                        raise SpeechFailed(
                            f"ElevenLabs refused the speech request ({response.status_code})"
                        )
                    async for chunk in response.aiter_bytes():
                        buffer += chunk
                        while len(buffer) >= FRAME_BYTES:
                            heard = True
                            yield frame(buffer[:FRAME_BYTES])
                            buffer = buffer[FRAME_BYTES:]
            except httpx.HTTPError as e:
                raise SpeechFailed(f"Could not reach ElevenLabs: {type(e).__name__}") from None
        if buffer:
            heard = True
            yield frame(buffer.ljust(FRAME_BYTES, b"\x00"))
        if not heard:
            raise SpeechFailed("ElevenLabs returned no audio")


def frame(pcm: bytes) -> rtc.AudioFrame:
    return rtc.AudioFrame(
        data=pcm, sample_rate=TTS_SAMPLE_RATE, num_channels=1, samples_per_channel=FRAME_SAMPLES
    )


SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def spoken_text(answer: str, limit: int) -> str:
    """The answer up to its last whole sentence within `limit` characters; a first sentence
    longer than that is cut after its last whole word, with an ellipsis."""
    answer = " ".join(answer.split())
    if len(answer) <= limit:
        return answer
    kept = ""
    for sentence in SENTENCE_END.split(answer):
        joined = f"{kept} {sentence}".strip()
        if len(joined) > limit:
            break
        kept = joined
    if kept:
        return kept
    words = answer[: limit + 1].split()
    if len(" ".join(words)) > limit:
        words = words[:-1]
    return " ".join(words or [answer[:limit]]) + "…"
