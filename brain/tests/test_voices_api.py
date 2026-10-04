"""Agent voices from ElevenLabs, or a clear unavailable state without a key."""

import httpx
import pytest
from api_support import ALEX

from brain.api.deps import get_http_transport

API_KEY = "el-test-key"


def voice(n: int, **fields) -> dict:
    return {
        "voice_id": f"v{n}",
        "name": f"Voice {n}",
        "description": f"Voice number {n}",
        "preview_url": f"https://storage.test/v{n}.mp3",
        "category": "premade",
        "labels": {"accent": "american", "gender": "female"},
        **fields,
    }


@pytest.fixture
def elevenlabs(app, settings):
    """Serve canned ElevenLabs pages; returns the requests it saw."""
    settings.elevenlabs_api_key = API_KEY
    seen: list[httpx.Request] = []
    pages = {
        None: {"voices": [voice(1), voice(2)], "has_more": True, "next_page_token": "p2"},
        "p2": {
            "voices": [voice(3, description=None, preview_url=None)],
            "has_more": False,
            "next_page_token": None,
        },
    }

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.headers.get("xi-api-key") != API_KEY:
            return httpx.Response(401, json={"detail": {"status": "invalid_api_key"}})
        return httpx.Response(200, json=pages[request.url.params.get("next_page_token")])

    app.dependency_overrides[get_http_transport] = lambda: httpx.MockTransport(handle)
    return seen


def test_voices_come_from_elevenlabs_across_pages(client_as, elevenlabs):
    response = client_as(ALEX).get("/voices")

    assert response.status_code == 200, response.text
    voices = response.json()
    assert [v["id"] for v in voices] == ["v1", "v2", "v3"]
    assert voices[0] == {
        "id": "v1",
        "name": "Voice 1",
        "desc": "Voice number 1",
        "sample": "https://storage.test/v1.mp3",
    }
    assert voices[2]["desc"] == "american, female"  # labels when there is no description
    assert voices[2]["sample"] == ""
    first = elevenlabs[0]
    assert first.method == "GET"
    assert first.url.path == "/v2/voices"
    assert first.url.params["page_size"] == "100"


def test_without_a_key_voices_are_unavailable(client_as, settings):
    settings.elevenlabs_api_key = None

    response = client_as(ALEX).get("/voices")

    assert response.status_code == 503
    assert "ELEVENLABS_API_KEY" in response.json()["detail"]


def test_a_rejected_key_is_a_bad_gateway_not_an_empty_list(client_as, elevenlabs, settings):
    settings.elevenlabs_api_key = "wrong"

    response = client_as(ALEX).get("/voices")

    assert response.status_code == 502
    assert "401" in response.json()["detail"]


def test_an_unreachable_elevenlabs_is_a_bad_gateway(app, client_as, settings):
    settings.elevenlabs_api_key = API_KEY

    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    app.dependency_overrides[get_http_transport] = lambda: httpx.MockTransport(refuse)

    response = client_as(ALEX).get("/voices")

    assert response.status_code == 502
    assert "ElevenLabs" in response.json()["detail"]
