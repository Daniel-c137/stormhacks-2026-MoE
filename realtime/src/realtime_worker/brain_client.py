import asyncio
from typing import Protocol

import httpx

from contracts import (
    Answer,
    ChatMessage,
    Invocation,
    InvokeRequest,
    InvokeResponse,
    SegmentsIngest,
    TranscriptSegment,
)

from .config import Settings


class BrainUnavailable(RuntimeError):
    """The brain is not configured or did not answer after every attempt."""


class BrainRejected(RuntimeError):
    """The brain refused the request. Retrying the same request will not help.

    Carries the status and at most a short string detail: FastAPI's 422 echoes the request,
    which for segments is transcript text, and this message ends up in logs."""

    def __init__(self, status: int, detail: str = ""):
        super().__init__(
            f"brain refused the request ({status})" + (f": {detail}" if detail else "")
        )
        self.status = status


# Transient by convention; everything else 4xx/501 is final. 501 is the brain's "not built yet".
RETRYABLE = {408, 429, 500, 502, 503, 504}
MAX_DETAIL = 200


class BrainClient(Protocol):
    """HTTP client for the brain's /internal routes."""

    async def ingest_segments(self, meeting_id: str, segments: list[TranscriptSegment]) -> None: ...

    async def ingest_public_chat(self, meeting_id: str, message: ChatMessage) -> None: ...

    async def invoke(self, invocation: Invocation, recent: list[TranscriptSegment]) -> Answer: ...


class HttpBrainClient:
    """Retries only idempotent calls: network failures and transient statuses, with backoff.
    Saving segments is idempotent (the brain ignores an identical seg_id); invoke is not, since
    a retry could run the model twice and make two response cards."""

    def __init__(
        self,
        base_url: str,
        token: str,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        attempts: int = 3,
        backoff: float = 0.5,
        timeout: float = 5.0,
        invoke_timeout: float = 30.0,
    ):
        self._http = httpx.AsyncClient(
            base_url=base_url,
            headers={"X-Internal-Token": token},
            transport=transport,
            timeout=timeout,
        )
        self._attempts = max(1, attempts)
        self._backoff = backoff
        self._invoke_timeout = invoke_timeout

    async def ingest_segments(self, meeting_id: str, segments: list[TranscriptSegment]) -> None:
        if not segments:
            return
        body = SegmentsIngest(segments=segments).model_dump(mode="json")
        await self._post(f"/internal/meetings/{meeting_id}/segments", body, idempotent=True)

    async def invoke(self, invocation: Invocation, recent: list[TranscriptSegment]) -> Answer:
        body = InvokeRequest(invocation=invocation, recent_segments=recent).model_dump(mode="json")
        response = await self._post(
            f"/internal/meetings/{invocation.meeting_id}/invoke",
            body,
            idempotent=False,
            timeout=self._invoke_timeout,
        )
        return InvokeResponse.model_validate_json(response.content).answer

    async def aclose(self) -> None:
        await self._http.aclose()

    async def _post(
        self, path: str, body: dict, *, idempotent: bool, timeout: float | None = None
    ) -> httpx.Response:
        attempts = self._attempts if idempotent else 1
        last = "no attempt made"
        for attempt in range(attempts):
            if attempt:
                await asyncio.sleep(self._backoff * 2 ** (attempt - 1))
            try:
                response = await self._http.post(
                    path, json=body, timeout=timeout or httpx.USE_CLIENT_DEFAULT
                )
            except httpx.TransportError as e:
                last = type(e).__name__
                continue
            if response.is_success:
                return response
            if response.status_code not in RETRYABLE:
                raise BrainRejected(response.status_code, short_detail(response))
            last = f"HTTP {response.status_code}"
        raise BrainUnavailable(f"POST {path} failed after {attempts} attempt(s) ({last})")


def short_detail(response: httpx.Response) -> str:
    """The brain's own string detail, capped. Validation errors (a list) are left out."""
    try:
        detail = response.json().get("detail")
    except (ValueError, AttributeError):
        return ""
    return detail[:MAX_DETAIL] if isinstance(detail, str) else ""


def brain_client_from_settings(settings: Settings) -> HttpBrainClient:
    required = {
        "BRAIN_URL": settings.brain_url,
        "BRAIN_INTERNAL_TOKEN": settings.brain_internal_token,
    }
    if missing := [name for name, value in required.items() if not value]:
        raise BrainUnavailable(f"The brain is not configured: set {' and '.join(missing)}")
    return HttpBrainClient(settings.brain_url, settings.brain_internal_token)
