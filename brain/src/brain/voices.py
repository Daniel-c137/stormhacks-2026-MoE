"""The ElevenLabs voices a team can pick for the agent."""

import httpx

from contracts import Voice

from .config import Settings

PAGE_SIZE = 100
MAX_PAGES = 10
TIMEOUT = httpx.Timeout(10.0, connect=5.0)


class VoicesUnavailable(RuntimeError):
    """ElevenLabs is not configured."""


class VoicesFailed(RuntimeError):
    """ElevenLabs is configured but the call failed."""


async def fetch_voices(
    config: Settings, *, transport: httpx.AsyncBaseTransport | None = None
) -> list[Voice]:
    """Every voice the API key can use (GET /v2/voices), page by page, in the order given."""
    if not config.elevenlabs_api_key:
        raise VoicesUnavailable("ElevenLabs is not configured: set ELEVENLABS_API_KEY")
    voices: list[Voice] = []
    params: dict[str, str | int] = {"page_size": PAGE_SIZE}
    async with httpx.AsyncClient(
        base_url=config.elevenlabs_api_url,
        headers={"xi-api-key": config.elevenlabs_api_key},
        timeout=TIMEOUT,
        transport=transport,
    ) as client:
        for _ in range(MAX_PAGES):
            try:
                response = await client.get("/v2/voices", params=params)
            except httpx.HTTPError as e:
                raise VoicesFailed(
                    f"Could not reach ElevenLabs: {str(e) or type(e).__name__}"
                ) from e
            if response.is_error:
                raise VoicesFailed(
                    f"ElevenLabs refused the voices request ({response.status_code})"
                )
            try:
                page = response.json()
            except ValueError as e:
                raise VoicesFailed("ElevenLabs returned something other than JSON") from e
            voices += [to_voice(v) for v in page.get("voices", [])]
            token = page.get("next_page_token")
            if not page.get("has_more") or not token:
                break
            params = {"page_size": PAGE_SIZE, "next_page_token": token}
    return voices


def to_voice(raw: dict) -> Voice:
    labels = raw.get("labels") or {}
    desc = raw.get("description") or ", ".join(str(v) for v in labels.values() if v)
    return Voice(
        id=raw["voice_id"],
        name=raw.get("name") or raw["voice_id"],
        desc=desc,
        sample=raw.get("preview_url") or "",
    )
