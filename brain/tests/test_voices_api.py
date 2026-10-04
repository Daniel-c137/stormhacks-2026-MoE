"""Agent voices from ElevenLabs, or a clear unavailable state without a key."""

import logging

import anyio
import httpx
import pytest
from api_support import ALEX, TEAM

from brain.api.deps import get_http_transport
from contracts import Identity, get_identity

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
        "default_label": None,
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


# the agent's default voice (ELEVENLABS_VOICE_ID)


def test_the_default_voice_comes_first_labelled_for_the_agent(client_as, elevenlabs, settings):
    settings.elevenlabs_voice_id = "v3"

    response = client_as(ALEX).get("/voices")

    assert response.status_code == 200, response.text
    voices = response.json()
    assert [v["id"] for v in voices] == ["v3", "v1", "v2"]
    assert voices[0]["name"] == "Voice 3"
    assert voices[0]["default_label"] == f"{get_identity().agent_name}'s default voice"
    assert [v["default_label"] for v in voices[1:]] == [None, None]


def test_the_default_voice_label_uses_the_identity(client_as, elevenlabs, settings, monkeypatch):
    monkeypatch.setattr(
        "brain.voices.get_identity", lambda: Identity(product_name="Acme", agent_name="Nova")
    )
    settings.elevenlabs_voice_id = "v2"

    voices = client_as(ALEX).get("/voices").json()

    assert voices[0]["id"] == "v2"
    assert voices[0]["default_label"] == "Nova's default voice"
    assert get_identity().agent_name not in str(voices)


def test_a_default_voice_missing_from_the_account_is_reported_not_fatal(
    client_as, elevenlabs, settings, caplog
):
    settings.elevenlabs_voice_id = "gone"

    with caplog.at_level(logging.WARNING, logger="brain.voices"):
        response = client_as(ALEX).get("/voices")

    assert response.status_code == 200, response.text
    voices = response.json()
    assert [v["id"] for v in voices] == ["v1", "v2", "v3"]
    assert all(v["default_label"] is None for v in voices)
    assert "ELEVENLABS_VOICE_ID" in caplog.text and "gone" in caplog.text
    assert API_KEY not in caplog.text


def test_without_a_default_voice_the_order_is_elevenlabs(client_as, elevenlabs, settings):
    settings.elevenlabs_voice_id = None

    voices = client_as(ALEX).get("/voices").json()

    assert [v["id"] for v in voices] == ["v1", "v2", "v3"]
    assert all(v["default_label"] is None for v in voices)


def test_a_teams_own_voice_does_not_change_the_default(client_as, elevenlabs, settings, store):
    async def pick():
        team = await store.settings(TEAM.id)
        await store.save_settings(team.model_copy(update={"voice": "v1"}))

    anyio.run(pick)
    settings.elevenlabs_voice_id = "v3"

    voices = client_as(ALEX).get("/voices").json()

    assert [v["id"] for v in voices] == ["v3", "v1", "v2"]
    assert voices[0]["default_label"] and voices[1]["default_label"] is None
