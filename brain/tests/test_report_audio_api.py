"""Listen: the report's summary read aloud, word for word, in the agent's voice (ElevenLabs).
Made once, then served from the store until the summary or the voice changes."""

import asyncio
import json
import logging

import anyio
import httpx
import pytest
from api_support import ALEX, OUTSIDER, SARAH, TEAM, create

from brain.api.deps import get_http_transport
from brain.store import NotFound
from contracts import Report

API_KEY = "el-test-key"
MODEL = "el-test-model"
VOICE = "default-voice"
TEAM_VOICE = "team-voice"
AUDIO = b"ID3\x04\x00\x00\x00\x00\x00\x00" + b"\xff\xfb\x90\x00" * 16
# Read as it is: markdown, odd spacing and punctuation are not cleaned up.
SUMMARY = "We ship **Friday**.\n\n  Bob owns the refund fix -- maybe?  Open: the waitlist email."


class FakeElevenLabs:
    """Text-to-speech that answers AUDIO, or `status` with an error body echoing the text."""

    def __init__(self):
        self.requests: list[httpx.Request] = []
        self.status = 200
        self.delay = 0.0

    async def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.delay:
            await anyio.sleep(self.delay)
        if request.headers.get("xi-api-key") != API_KEY:
            return httpx.Response(401, json={"detail": {"status": "invalid_api_key"}})
        if self.status != 200:
            text = json.loads(request.content)["text"]
            return httpx.Response(self.status, json={"detail": {"message": f"failed: {text}"}})
        return httpx.Response(200, content=AUDIO, headers={"content-type": "audio/mpeg"})


@pytest.fixture
def elevenlabs(app, settings) -> FakeElevenLabs:
    settings.elevenlabs_api_key = API_KEY
    settings.elevenlabs_tts_model = MODEL
    settings.elevenlabs_voice_id = VOICE
    fake = FakeElevenLabs()
    app.dependency_overrides[get_http_transport] = lambda: httpx.MockTransport(fake.handle)
    return fake


def reported(store, meeting: dict, summary: str = SUMMARY, status: str = "needs_review") -> None:
    """Save the meeting's report with this summary, as the write-up would."""

    async def save():
        await store.save_report(Report(meeting_id=meeting["id"], summary=summary))
        await store.set_status(meeting["id"], status)

    asyncio.run(save())


def team_voice(store, voice: str | None) -> None:
    async def save():
        settings = await store.settings(TEAM.id)
        await store.save_settings(settings.model_copy(update={"voice": voice}))

    asyncio.run(save())


def stored(store, meeting: dict):
    return asyncio.run(store.report_audio(meeting["id"]))


def listen(client, meeting: dict) -> httpx.Response:
    return client.get(f"/meetings/{meeting['id']}/report/audio")


def test_the_first_listen_reads_the_summary_verbatim_and_stores_the_audio(
    client_as, store, elevenlabs
):
    meeting = create(client_as(ALEX))
    reported(store, meeting)

    response = listen(client_as(SARAH), meeting)

    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "audio/mpeg"
    assert response.content == AUDIO
    [request] = elevenlabs.requests
    assert request.method == "POST"
    assert request.url.path == f"/v1/text-to-speech/{VOICE}"
    assert request.url.params["output_format"].startswith("mp3_")
    assert request.headers["xi-api-key"] == API_KEY
    assert json.loads(request.content) == {"text": SUMMARY, "model_id": MODEL}
    saved = stored(store, meeting)
    assert (saved.content_type, saved.data) == ("audio/mpeg", AUDIO)


def test_a_second_listen_serves_the_stored_audio_without_calling_elevenlabs(
    client_as, store, elevenlabs
):
    meeting = create(client_as(ALEX))
    reported(store, meeting)
    assert listen(client_as(ALEX), meeting).status_code == 200
    elevenlabs.status = 500  # would fail if it were called again

    response = listen(client_as(SARAH), meeting)

    assert response.status_code == 200, response.text
    assert response.content == AUDIO
    assert response.headers["content-type"] == "audio/mpeg"
    assert len(elevenlabs.requests) == 1


def test_the_teams_voice_overrides_the_default(client_as, store, elevenlabs):
    meeting = create(client_as(ALEX))
    reported(store, meeting)
    team_voice(store, TEAM_VOICE)

    assert listen(client_as(ALEX), meeting).status_code == 200

    [request] = elevenlabs.requests
    assert request.url.path == f"/v1/text-to-speech/{TEAM_VOICE}"


def test_a_team_voice_is_enough_without_a_default_voice(client_as, store, elevenlabs, settings):
    settings.elevenlabs_voice_id = None
    meeting = create(client_as(ALEX))
    reported(store, meeting)
    team_voice(store, TEAM_VOICE)

    assert listen(client_as(ALEX), meeting).status_code == 200
    assert elevenlabs.requests[0].url.path == f"/v1/text-to-speech/{TEAM_VOICE}"


def test_a_changed_summary_is_read_again(client_as, store, elevenlabs):
    meeting = create(client_as(ALEX))
    reported(store, meeting)
    assert listen(client_as(ALEX), meeting).status_code == 200
    reported(store, meeting, summary="We ship Monday instead.")

    response = listen(client_as(ALEX), meeting)

    assert response.status_code == 200
    assert len(elevenlabs.requests) == 2
    texts = [json.loads(r.content)["text"] for r in elevenlabs.requests]
    assert texts == [SUMMARY, "We ship Monday instead."]
    assert listen(client_as(ALEX), meeting).status_code == 200
    assert len(elevenlabs.requests) == 2  # the new audio is stored and served


def test_a_changed_voice_is_read_again(client_as, store, elevenlabs):
    meeting = create(client_as(ALEX))
    reported(store, meeting)
    assert listen(client_as(ALEX), meeting).status_code == 200
    team_voice(store, TEAM_VOICE)

    assert listen(client_as(ALEX), meeting).status_code == 200

    paths = [r.url.path for r in elevenlabs.requests]
    assert paths == [f"/v1/text-to-speech/{VOICE}", f"/v1/text-to-speech/{TEAM_VOICE}"]


@pytest.mark.parametrize(
    ("setting", "env"),
    [
        ("elevenlabs_api_key", "ELEVENLABS_API_KEY"),
        ("elevenlabs_tts_model", "ELEVENLABS_TTS_MODEL"),
        ("elevenlabs_voice_id", "ELEVENLABS_VOICE_ID"),
    ],
)
def test_each_missing_piece_of_elevenlabs_is_unavailable(
    client_as, store, elevenlabs, settings, setting, env
):
    setattr(settings, setting, None)
    meeting = create(client_as(ALEX))
    reported(store, meeting)

    response = listen(client_as(ALEX), meeting)

    assert response.status_code == 503
    assert env in response.json()["detail"]
    assert elevenlabs.requests == []
    with pytest.raises(NotFound):
        stored(store, meeting)


def test_an_elevenlabs_error_is_a_bad_gateway_and_stores_nothing(
    client_as, store, elevenlabs, caplog
):
    caplog.set_level(logging.DEBUG)
    meeting = create(client_as(ALEX))
    reported(store, meeting)
    elevenlabs.status = 500

    response = listen(client_as(ALEX), meeting)

    assert response.status_code == 502
    assert "ElevenLabs" in response.json()["detail"]
    assert "Friday" not in response.text  # the summary is never echoed...
    assert "Friday" not in caplog.text  # ...or logged
    with pytest.raises(NotFound):
        stored(store, meeting)


def test_an_unreachable_elevenlabs_is_a_bad_gateway_and_stores_nothing(
    app, client_as, store, settings
):
    settings.elevenlabs_api_key = API_KEY
    settings.elevenlabs_tts_model = MODEL
    settings.elevenlabs_voice_id = VOICE

    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    app.dependency_overrides[get_http_transport] = lambda: httpx.MockTransport(refuse)
    meeting = create(client_as(ALEX))
    reported(store, meeting)

    response = listen(client_as(ALEX), meeting)

    assert response.status_code == 502
    assert "ElevenLabs" in response.json()["detail"]
    with pytest.raises(NotFound):
        stored(store, meeting)


def test_no_report_yet_is_not_found_like_the_report(client_as, elevenlabs):
    meeting = create(client_as(ALEX))

    response = listen(client_as(ALEX), meeting)
    report = client_as(ALEX).get(f"/meetings/{meeting['id']}/report")

    assert response.status_code == report.status_code == 404
    assert response.json()["detail"] == report.json()["detail"]
    assert elevenlabs.requests == []


def test_a_write_up_still_running_is_a_conflict(client_as, store, elevenlabs):
    meeting = create(client_as(ALEX))
    reported(store, meeting, status="processing")  # a retry is rewriting the report

    response = listen(client_as(ALEX), meeting)

    assert response.status_code == 409
    assert "still running" in response.json()["detail"]
    assert elevenlabs.requests == []


@pytest.mark.parametrize("summary", ["", "  \n "])
def test_an_empty_summary_has_nothing_to_read(client_as, store, elevenlabs, summary):
    meeting = create(client_as(ALEX))
    reported(store, meeting, summary=summary)

    response = listen(client_as(ALEX), meeting)

    assert response.status_code == 404
    assert "nothing to read" in response.json()["detail"]
    assert elevenlabs.requests == []


def test_another_teams_meeting_is_not_found(client_as, store, elevenlabs):
    meeting = create(client_as(ALEX))
    reported(store, meeting)

    response = listen(client_as(OUTSIDER), meeting)

    assert response.status_code == 404
    assert response.json()["detail"] == "Meeting not found"
    assert elevenlabs.requests == []


@pytest.mark.anyio
async def test_listens_at_the_same_time_call_elevenlabs_once(app, client_as, store, elevenlabs):
    meeting = await anyio.to_thread.run_sync(create, client_as(ALEX))
    await store.save_report(Report(meeting_id=meeting["id"], summary=SUMMARY))
    await store.set_status(meeting["id"], "needs_review")
    elevenlabs.delay = 0.2
    responses: list[httpx.Response] = []

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://brain.test",
        headers={"X-Test-User": ALEX.id},
    ) as client:

        async def one():
            responses.append(await client.get(f"/meetings/{meeting['id']}/report/audio"))

        async with anyio.create_task_group() as group:
            for _ in range(3):
                group.start_soon(one)

    assert [r.status_code for r in responses] == [200, 200, 200]
    assert all(r.content == AUDIO for r in responses)
    assert len(elevenlabs.requests) == 1
