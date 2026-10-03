"""The worker's HTTP client for the brain's /internal routes."""

import httpx
import pytest

from contracts import TranscriptSegment
from realtime_worker.brain_client import (
    BrainRejected,
    BrainUnavailable,
    HttpBrainClient,
    brain_client_from_settings,
)
from realtime_worker.config import Settings

pytestmark = pytest.mark.anyio

TOKEN = "worker-shared-secret"


def seg(meeting_id: str, n: int) -> TranscriptSegment:
    return TranscriptSegment(
        seg_id=f"{meeting_id}-u-alex-{n}",
        meeting_id=meeting_id,
        speaker_id="u-alex",
        speaker_name="Alex Chen",
        text=f"thing {n}",
        is_final=True,
        t_start=float(n),
        t_end=float(n) + 1,
    )


class Recorder:
    """httpx transport that answers with scripted statuses (or raises) and records requests."""

    def __init__(self, *responses: int | Exception):
        self.responses = list(responses)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        outcome = self.responses.pop(0) if self.responses else 204
        if isinstance(outcome, Exception):
            raise outcome
        return httpx.Response(outcome)


def client(recorder: Recorder, attempts: int = 3) -> HttpBrainClient:
    return HttpBrainClient(
        "http://brain.test",
        TOKEN,
        transport=httpx.MockTransport(recorder),
        attempts=attempts,
        backoff=0,
    )


async def test_posts_segments_to_the_internal_route_with_the_worker_token():
    recorder = Recorder(204)

    await client(recorder).ingest_segments("m-1", [seg("m-1", 1), seg("m-1", 2)])

    [request] = recorder.requests
    assert request.method == "POST"
    assert str(request.url) == "http://brain.test/internal/meetings/m-1/segments"
    assert request.headers["X-Internal-Token"] == TOKEN
    body = httpx.Response(200, content=request.content).json()
    assert [s["seg_id"] for s in body["segments"]] == ["m-1-u-alex-1", "m-1-u-alex-2"]


async def test_retries_server_errors_and_network_failures_then_succeeds():
    recorder = Recorder(503, httpx.ConnectError("refused"), 204)

    await client(recorder).ingest_segments("m-1", [seg("m-1", 1)])

    assert len(recorder.requests) == 3


async def test_gives_up_after_the_last_attempt():
    recorder = Recorder(500, 502, 503)

    with pytest.raises(BrainUnavailable):
        await client(recorder, attempts=3).ingest_segments("m-1", [seg("m-1", 1)])

    assert len(recorder.requests) == 3


@pytest.mark.parametrize("status", [401, 404, 422])
async def test_a_rejection_is_not_retried(status):
    recorder = Recorder(status)

    with pytest.raises(BrainRejected) as caught:
        await client(recorder).ingest_segments("m-1", [seg("m-1", 1)])

    assert caught.value.status == status
    assert len(recorder.requests) == 1


async def test_nothing_to_send_makes_no_request():
    recorder = Recorder()

    await client(recorder).ingest_segments("m-1", [])

    assert recorder.requests == []


def test_missing_brain_settings_are_reported_not_guessed():
    with pytest.raises(BrainUnavailable, match="BRAIN_URL"):
        brain_client_from_settings(Settings(_env_file=None, brain_internal_token=TOKEN))
    with pytest.raises(BrainUnavailable, match="BRAIN_INTERNAL_TOKEN"):
        brain_client_from_settings(Settings(_env_file=None, brain_url="http://brain.test"))


# contract with the brain's real endpoint


async def test_segments_land_in_the_real_brain_transcript():
    from brain.api.deps import get_settings, get_store
    from brain.config import Settings as BrainSettings
    from brain.main import create_app
    from brain.store import InMemoryStore
    from contracts import Team

    store = InMemoryStore(teams=[Team(id="t-1", name="Checkout", member_ids=["u-alex"])], people=[])
    meeting = await store.create_meeting("t-1", "Standup", host_id="u-alex")
    app = create_app()
    app.dependency_overrides[get_store] = lambda: store
    app.dependency_overrides[get_settings] = lambda: BrainSettings(
        _env_file=None, brain_internal_token=TOKEN
    )
    brain = HttpBrainClient(
        "http://brain.test", TOKEN, transport=httpx.ASGITransport(app=app), backoff=0
    )

    await brain.ingest_segments(meeting.id, [seg(meeting.id, 1), seg(meeting.id, 2)])
    await brain.ingest_segments(meeting.id, [seg(meeting.id, 2)])  # a retry after a timeout

    saved = await store.transcript(meeting.id)
    assert [s.seg_id for s in saved] == [seg(meeting.id, 1).seg_id, seg(meeting.id, 2).seg_id]
