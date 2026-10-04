"""ElevenLabs Scribe v2 Realtime as the worker's SpeechToText: one websocket session per stream.

Audio goes up as 16-bit 16 kHz mono PCM in 0.1 s base64 chunks (wss .../v1/speech-to-text/
realtime, audio_format=pcm_16000). Scribe's voice activity detection commits each utterance
(commit_strategy=vad): partial_transcript messages become partial pieces and each
committed_transcript_with_timestamps becomes the final piece, timed by its words. Scribe sends
every commit twice; the copy without timestamps is ignored. When the input ends the last utterance
is committed by hand and its final awaited; if none comes, the last partial is kept as the final.

Unless a language is configured, Scribe detects each utterance's language
(include_language_detection) and the committed message carries it as language_code; partials
don't, so a partial piece's language is None (#106).
"""

import asyncio
import base64
import json
import logging
import time
import unicodedata
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable
from dataclasses import dataclass
from typing import Literal, Protocol
from urllib.parse import urlencode

import numpy as np
from livekit import rtc
from websockets.exceptions import ConnectionClosed

from contracts.language import normalise_language

from .stt import SpeechPiece

log = logging.getLogger(__name__)

SCRIBE_SAMPLE_RATE = 16000
CHUNK_BYTES = SCRIBE_SAMPLE_RATE * 2 // 10  # 0.1 s of 16-bit mono
ERRORS = frozenset(
    {
        "error",
        "auth_error",
        "quota_exceeded",
        "commit_throttled",
        "unaccepted_terms",
        "rate_limited",
        "queue_overflow",
        "resource_exhausted",
        "session_time_limit_exceeded",
        "input_error",
        "invalid_request",
        "chunk_size_exceeded",
        "insufficient_audio_activity",
        "transcriber_error",
    }
)


# Scribe has returned full-width commas and question marks for English speech, which hide
# the wake phrase's comma and the question mark from everything downstream.
FULL_WIDTH = str.maketrans(
    {"\uff0c": ", ", "\u3002": ". ", "\uff1f": "? ", "\uff01": "! ", "\uff1a": ": ", "\uff1b": "; "}
)


def normalise(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text.translate(FULL_WIDTH)).split())


class ScribeError(RuntimeError):
    """The session failed or dropped. The TranscriptionManager reopens it."""


class Connection(Protocol):
    async def send(self, message: str) -> None: ...

    async def recv(self) -> str | bytes: ...

    async def close(self) -> None: ...


Connect = Callable[[str, dict[str, str]], Awaitable[Connection]]


async def websocket_connect(url: str, headers: dict[str, str]) -> Connection:
    from websockets.asyncio.client import connect

    return await connect(url, additional_headers=headers, open_timeout=10)


@dataclass(frozen=True)
class ScribeEvent:
    kind: Literal["partial", "final", "error", "warning", "ignore"]
    text: str = ""
    start: float | None = None
    end: float | None = None
    error: str = ""
    language: str | None = None


def parse_event(raw: str | bytes) -> ScribeEvent:
    try:
        message = json.loads(raw)
        kind = message.get("message_type")
    except (ValueError, AttributeError):
        return ScribeEvent("ignore")
    if kind == "partial_transcript":
        return ScribeEvent("partial", text=normalise(str(message.get("text") or "")))
    if kind == "committed_transcript_with_timestamps":
        words = [w for w in message.get("words") or [] if w.get("type", "word") == "word"]
        start = words[0].get("start") if words else None
        end = words[-1].get("end") if words else None
        text = normalise(str(message.get("text") or ""))
        language = normalise_language(message.get("language_code"))
        return ScribeEvent("final", text=text, start=start, end=end, language=language)
    if kind in ERRORS:
        return ScribeEvent(
            "error", error=f"{kind}: {message.get('error') or message.get('message')}"
        )
    if kind == "warning":
        return ScribeEvent("warning", error=str(message.get("error") or message.get("message")))
    return ScribeEvent("ignore")  # session_started, committed_transcript, entities, ...


class Pcm16kMono:
    """LiveKit frames in any rate and channel count -> 16-bit 16 kHz mono, in 0.1 s chunks."""

    def __init__(self):
        self._resampler: rtc.AudioResampler | None = None
        self._rate: int | None = None
        self._buffer = b""

    def push(self, frame: rtc.AudioFrame) -> list[bytes]:
        samples = np.frombuffer(frame.data, dtype=np.int16)
        if frame.num_channels > 1:
            samples = samples.reshape(-1, frame.num_channels).mean(axis=1).astype(np.int16)
        if frame.sample_rate == SCRIBE_SAMPLE_RATE:
            self._buffer += samples.tobytes()
        else:
            if self._rate != frame.sample_rate:
                self._buffer += self._drain()
                self._resampler = rtc.AudioResampler(frame.sample_rate, SCRIBE_SAMPLE_RATE)
                self._rate = frame.sample_rate
            mono = rtc.AudioFrame(samples.tobytes(), frame.sample_rate, 1, len(samples))
            for out in self._resampler.push(mono):
                self._buffer += bytes(out.data)
        return self._chunks()

    def flush(self) -> list[bytes]:
        self._buffer += self._drain()
        chunks = self._chunks()
        if self._buffer:
            chunks.append(self._buffer)
            self._buffer = b""
        return chunks

    def _drain(self) -> bytes:
        if self._resampler is None:
            return b""
        tail = b"".join(bytes(out.data) for out in self._resampler.flush())
        self._resampler, self._rate = None, None
        return tail

    def _chunks(self) -> list[bytes]:
        chunks = []
        while len(self._buffer) >= CHUNK_BYTES:
            chunks.append(self._buffer[:CHUNK_BYTES])
            self._buffer = self._buffer[CHUNK_BYTES:]
        return chunks


class ScribeSTT:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        keyterms: Iterable[str] = (),
        language: str | None = None,
        url: str = "https://api.elevenlabs.io",
        connect: Connect = websocket_connect,
        finalize_seconds: float = 3.0,
        keepalive_seconds: float = 10.0,
    ):
        self._api_key = api_key
        self._model = model
        self._keyterms = list(keyterms)
        self._language = language
        self._url = url
        self._connect = connect
        self._finalize_seconds = finalize_seconds
        self._keepalive_seconds = keepalive_seconds

    def realtime_url(self) -> str:
        base = self._url.rstrip("/").replace("https://", "wss://", 1).replace("http://", "ws://", 1)
        query = [
            ("model_id", self._model),
            ("audio_format", f"pcm_{SCRIBE_SAMPLE_RATE}"),
            ("commit_strategy", "vad"),
            ("include_timestamps", "true"),
            *(
                (("language_code", self._language),)
                if self._language
                else (("include_language_detection", "true"),)
            ),
            *(("keyterms", term) for term in self._keyterms),
        ]
        return f"{base}/v1/speech-to-text/realtime?{urlencode(query)}"

    async def stream(self, audio: AsyncIterator[rtc.AudioFrame]) -> AsyncIterator[SpeechPiece]:
        ws = await self._connect(self.realtime_url(), {"xi-api-key": self._api_key})
        session = _Session(ws, self._keepalive_seconds)
        tasks = [
            asyncio.create_task(session.send_audio(audio)),
            asyncio.create_task(session.receive()),
            asyncio.create_task(session.keepalive()),
        ]
        try:
            async for piece in session.pieces(self._finalize_seconds):
                yield piece
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            try:
                await ws.close()
            except Exception:
                log.debug("Closing the Scribe websocket failed", exc_info=True)
            log.info("Scribe stream closed after %.1f s of audio", session.sent_seconds)


class _Session:
    """One websocket: a sender, a receiver and a keepalive feed one queue of what happened."""

    def __init__(self, ws: Connection, keepalive_seconds: float):
        self._ws = ws
        self._keepalive_seconds = keepalive_seconds
        self._events: asyncio.Queue[tuple[str, object]] = asyncio.Queue()
        self._send_lock = asyncio.Lock()
        self._last_send = time.monotonic()
        self.sent_seconds = 0.0

    async def _send(self, pcm: bytes, *, commit: bool = False) -> None:
        message = {
            "message_type": "input_audio_chunk",
            "audio_base_64": base64.b64encode(pcm).decode(),
            "commit": commit,
            "sample_rate": SCRIBE_SAMPLE_RATE,
        }
        async with self._send_lock:
            await self._ws.send(json.dumps(message))
            self._last_send = time.monotonic()
        self.sent_seconds += len(pcm) / (SCRIBE_SAMPLE_RATE * 2)

    async def send_audio(self, audio: AsyncIterator[rtc.AudioFrame]) -> None:
        pcm = Pcm16kMono()
        try:
            async for frame in audio:
                for chunk in pcm.push(frame):
                    await self._send(chunk)
            for chunk in pcm.flush():
                await self._send(chunk)
            await self._send(b"", commit=True)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self._events.put_nowait(("failed", e))
            return
        self._events.put_nowait(("ended", None))

    async def receive(self) -> None:
        try:
            while True:
                self._events.put_nowait(("message", await self._ws.recv()))
        except ConnectionClosed as e:
            self._events.put_nowait(("closed", e))
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self._events.put_nowait(("failed", e))

    async def keepalive(self) -> None:
        """Scribe closes a session that hears nothing; a quiet track sends empty chunks."""
        while True:
            wait = self._last_send + self._keepalive_seconds - time.monotonic()
            if wait > 0:
                await asyncio.sleep(wait)
                continue
            await self._send(b"")

    async def pieces(self, finalize_seconds: float) -> AsyncIterator[SpeechPiece]:
        utterance_start = 0.0
        partial = ""
        deadline: float | None = None  # set once the input ended
        while True:
            timeout = None if deadline is None else max(0.0, deadline - time.monotonic())
            try:
                kind, value = await asyncio.wait_for(self._events.get(), timeout)
            except TimeoutError:
                break
            if kind == "ended":
                deadline = time.monotonic() + finalize_seconds
                continue
            if kind == "failed":
                raise ScribeError(f"Scribe session failed: {type(value).__name__}") from None
            if kind == "closed":
                if deadline is not None:
                    break
                raise ScribeError("Scribe closed the connection") from None
            event = parse_event(value)  # type: ignore[arg-type]
            if event.kind == "error":
                if deadline is not None:
                    log.info("Scribe at the end of the input: %s", event.error)
                    break
                raise ScribeError(f"Scribe reported {event.error}")
            if event.kind == "warning":
                log.info("Scribe warning: %s", event.error)
            elif event.kind == "partial":
                text = event.text.strip()
                if text and text != partial:
                    partial = text
                    yield SpeechPiece(text, False, utterance_start, self.sent_seconds)
            elif event.kind == "final":
                text = event.text.strip()
                partial = ""
                if text:
                    start = event.start if event.start is not None else utterance_start
                    end = event.end if event.end is not None else self.sent_seconds
                    yield SpeechPiece(text, True, start, max(start, end), event.language)
                    utterance_start = max(start, end)
                if deadline is not None:
                    break
        if partial:  # Scribe never committed the last words: keep them
            yield SpeechPiece(
                partial, True, utterance_start, max(utterance_start, self.sent_seconds)
            )
