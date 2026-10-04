import asyncio
import logging
from typing import Protocol

import httpx

from contracts import (
    AgendaTrackResponse,
    Answer,
    CatchUpRequest,
    CatchUpResponse,
    ChatMessage,
    FactCheckResponse,
    Invocation,
    InvokeRequest,
    InvokeResponse,
    KeytermsResponse,
    SegmentsIngest,
    TranscriptSegment,
    TranslateRequest,
    TranslateResponse,
    WorkerMeetingResponse,
)

from .config import Settings

log = logging.getLogger(__name__)


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
# An agenda tick whose model call failed still carries the agenda and the nudges it saved.
AGENDA_FALLBACK = {502, 503}


class BrainClient(Protocol):
    """HTTP client for the brain's /internal routes."""

    async def ingest_segments(self, meeting_id: str, segments: list[TranscriptSegment]) -> None: ...

    async def ingest_public_chat(self, meeting_id: str, message: ChatMessage) -> None: ...

    async def invoke(self, invocation: Invocation, recent: list[TranscriptSegment]) -> Answer: ...

    async def meeting(self, meeting_id: str) -> WorkerMeetingResponse: ...

    async def agent_joined(self, meeting_id: str) -> None: ...

    async def keyterms(self, meeting_id: str) -> list[str]: ...

    async def track_agenda(self, meeting_id: str) -> AgendaTrackResponse: ...

    async def fact_check(self, meeting_id: str) -> FactCheckResponse: ...

    async def catch_up(
        self, meeting_id: str, participant_id: str, since: float, until: float
    ) -> CatchUpResponse: ...

    async def translate(
        self, meeting_id: str, text: str, language: str | None
    ) -> TranslateResponse: ...


class HttpBrainClient:
    """Retries only idempotent calls: network failures and transient statuses, with backoff.
    Reads and saves are idempotent (the brain ignores an identical seg_id or chat id); invoke is
    not, since a retry could run the model twice and make two response cards, and a tick is not
    retried within itself because the next tick tries again."""

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
        translate_timeout: float = 4.0,
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
        self._translate_timeout = translate_timeout

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

    async def ingest_public_chat(self, meeting_id: str, message: ChatMessage) -> None:
        """Public chat only; a private message is refused here and never sent."""
        if message.visibility != "public" or message.recipient_id is not None:
            raise ValueError("Private chat is never sent to the brain")
        body = message.model_dump(mode="json")
        await self._post(f"/internal/meetings/{meeting_id}/chat", body, idempotent=True)

    async def meeting(self, meeting_id: str) -> WorkerMeetingResponse:
        response = await self._request("GET", f"/internal/meetings/{meeting_id}", idempotent=True)
        return WorkerMeetingResponse.model_validate_json(response.content)

    async def agent_joined(self, meeting_id: str) -> None:
        """Retried: the brain keeps the first join time, so a resend changes nothing."""
        await self._post(f"/internal/meetings/{meeting_id}/agent-joined", {}, idempotent=True)

    async def keyterms(self, meeting_id: str) -> list[str]:
        response = await self._request(
            "GET", f"/internal/meetings/{meeting_id}/keyterms", idempotent=True
        )
        return KeytermsResponse.model_validate_json(response.content).terms

    async def track_agenda(self, meeting_id: str) -> AgendaTrackResponse:
        """The brain takes `now` from the meeting's start. When the model failed (502/503) the
        body still holds the agenda and the nudges it saved as sent, so they are returned."""
        response = await self._request(
            "POST",
            f"/internal/meetings/{meeting_id}/agenda/track",
            body={},
            idempotent=False,
            timeout=self._invoke_timeout,
            keep=AGENDA_FALLBACK,
        )
        if response.status_code in AGENDA_FALLBACK:
            try:
                tracked = AgendaTrackResponse.model_validate_json(response.content)
            except ValueError:
                raise BrainUnavailable(
                    f"agenda tick failed (HTTP {response.status_code})"
                ) from None
            log.warning("Agenda tick tracked nothing: %s", short_detail(response))
            return tracked
        return AgendaTrackResponse.model_validate_json(response.content)

    async def fact_check(self, meeting_id: str) -> FactCheckResponse:
        response = await self._post(
            f"/internal/meetings/{meeting_id}/fact-check",
            {},
            idempotent=False,
            timeout=self._invoke_timeout,
        )
        return FactCheckResponse.model_validate_json(response.content)

    async def catch_up(
        self, meeting_id: str, participant_id: str, since: float, until: float
    ) -> CatchUpResponse:
        """What to send the participant about the span they missed. Not retried: each attempt
        is a model call, and a missed catch-up only costs them the summary."""
        body = CatchUpRequest(participant_id=participant_id, since=since, until=until)
        response = await self._post(
            f"/internal/meetings/{meeting_id}/catch-up",
            body.model_dump(mode="json"),
            idempotent=False,
            timeout=self._invoke_timeout,
        )
        return CatchUpResponse.model_validate_json(response.content)

    async def translate(
        self, meeting_id: str, text: str, language: str | None
    ) -> TranslateResponse:
        """Speech into English for a live caption (#106). One attempt with a short timeout: a
        caption that arrives late is worse than the original shown untranslated."""
        body = TranslateRequest(text=text, language=language).model_dump(mode="json")
        response = await self._post(
            f"/internal/meetings/{meeting_id}/translate",
            body,
            idempotent=False,
            timeout=self._translate_timeout,
        )
        return TranslateResponse.model_validate_json(response.content)

    async def aclose(self) -> None:
        await self._http.aclose()

    async def _post(
        self, path: str, body: dict, *, idempotent: bool, timeout: float | None = None
    ) -> httpx.Response:
        return await self._request("POST", path, body=body, idempotent=idempotent, timeout=timeout)

    async def _request(
        self,
        method: str,
        path: str,
        *,
        body: dict | None = None,
        idempotent: bool,
        timeout: float | None = None,
        keep: frozenset[int] | set[int] = frozenset(),
    ) -> httpx.Response:
        """The response on success, or on a status in `keep`."""
        attempts = self._attempts if idempotent else 1
        last = "no attempt made"
        for attempt in range(attempts):
            if attempt:
                await asyncio.sleep(self._backoff * 2 ** (attempt - 1))
            try:
                response = await self._http.request(
                    method, path, json=body, timeout=timeout or httpx.USE_CLIENT_DEFAULT
                )
            except httpx.TransportError as e:
                last = type(e).__name__
                continue
            if response.is_success or response.status_code in keep:
                return response
            if response.status_code not in RETRYABLE:
                raise BrainRejected(response.status_code, short_detail(response))
            last = f"HTTP {response.status_code}"
        raise BrainUnavailable(f"{method} {path} failed after {attempts} attempt(s) ({last})")


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
