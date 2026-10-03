import asyncio
from typing import Protocol

import httpx

from contracts import Answer, ChatMessage, Invocation, SegmentsIngest, TranscriptSegment

from .config import Settings


class BrainUnavailable(RuntimeError):
    """The brain is not configured or did not answer after every attempt."""


class BrainRejected(RuntimeError):
    """The brain refused the request (4xx). Retrying the same request will not help."""

    def __init__(self, status: int, detail: str):
        super().__init__(f"brain refused the request ({status}): {detail}")
        self.status = status


class BrainClient(Protocol):
    """HTTP client for the brain's /internal routes."""

    async def ingest_segments(self, meeting_id: str, segments: list[TranscriptSegment]) -> None: ...

    async def ingest_public_chat(self, meeting_id: str, message: ChatMessage) -> None: ...

    async def invoke(self, invocation: Invocation, recent: list[TranscriptSegment]) -> Answer: ...


class HttpBrainClient:
    """Retries network failures and 5xx with backoff; never retries a 4xx. Resending segments
    is safe because the brain ignores a seg_id it already saved."""

    def __init__(
        self,
        base_url: str,
        token: str,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        attempts: int = 3,
        backoff: float = 0.5,
        timeout: float = 5.0,
    ):
        self._http = httpx.AsyncClient(
            base_url=base_url,
            headers={"X-Internal-Token": token},
            transport=transport,
            timeout=timeout,
        )
        self._attempts = max(1, attempts)
        self._backoff = backoff

    async def ingest_segments(self, meeting_id: str, segments: list[TranscriptSegment]) -> None:
        if not segments:
            return
        body = SegmentsIngest(segments=segments).model_dump(mode="json")
        await self._post(f"/internal/meetings/{meeting_id}/segments", body)

    async def aclose(self) -> None:
        await self._http.aclose()

    async def _post(self, path: str, body: dict) -> httpx.Response:
        last = "no attempt made"
        for attempt in range(self._attempts):
            if attempt:
                await asyncio.sleep(self._backoff * 2 ** (attempt - 1))
            try:
                response = await self._http.post(path, json=body)
            except httpx.TransportError as e:
                last = f"{type(e).__name__}: {e}"
                continue
            if response.status_code >= 500:
                last = f"HTTP {response.status_code}"
                continue
            if response.status_code >= 400:
                raise BrainRejected(response.status_code, response.text)
            return response
        raise BrainUnavailable(f"POST {path} failed after {self._attempts} attempts ({last})")


def brain_client_from_settings(settings: Settings) -> HttpBrainClient:
    required = {
        "BRAIN_URL": settings.brain_url,
        "BRAIN_INTERNAL_TOKEN": settings.brain_internal_token,
    }
    if missing := [name for name, value in required.items() if not value]:
        raise BrainUnavailable(f"The brain is not configured: set {' and '.join(missing)}")
    return HttpBrainClient(settings.brain_url, settings.brain_internal_token)
