"""The worker's HTTP client for the brain's /internal routes."""

import json
from datetime import UTC, datetime

import httpx
import pytest

from contracts import (
    Agenda,
    AgendaNudge,
    AgendaTrackResponse,
    Answer,
    ChatMessage,
    FactCheck,
    FactCheckResponse,
    Invocation,
    Meeting,
    TranscriptSegment,
    TranslateResponse,
    WorkerMeetingResponse,
)
from realtime_worker.brain_client import (
    BrainRejected,
    BrainUnavailable,
    HttpBrainClient,
    brain_client_from_settings,
)
from realtime_worker.config import Settings

pytestmark = pytest.mark.anyio

TOKEN = "worker-shared-secret-0123456789abcdef"


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

    def __init__(self, *responses: int | httpx.Response | Exception):
        self.responses = list(responses)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        outcome = self.responses.pop(0) if self.responses else 204
        if isinstance(outcome, Exception):
            raise outcome
        return outcome if isinstance(outcome, httpx.Response) else httpx.Response(outcome)


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


@pytest.mark.parametrize("status", [401, 404, 409, 422, 501])
async def test_a_rejection_is_not_retried(status):
    recorder = Recorder(status)

    with pytest.raises(BrainRejected) as caught:
        await client(recorder).ingest_segments("m-1", [seg("m-1", 1)])

    assert caught.value.status == status
    assert len(recorder.requests) == 1


@pytest.mark.parametrize("status", [408, 429])
async def test_timeouts_and_rate_limits_are_retried(status):
    recorder = Recorder(status, 204)

    await client(recorder).ingest_segments("m-1", [seg("m-1", 1)])

    assert len(recorder.requests) == 2


async def test_a_rejection_never_carries_transcript_text_into_logs():
    echo = httpx.Response(
        422, json={"detail": [{"msg": "Field required", "input": {"text": "secret words"}}]}
    )

    with pytest.raises(BrainRejected) as caught:
        await client(Recorder(echo)).ingest_segments("m-1", [seg("m-1", 1)])

    assert "secret words" not in str(caught.value)
    assert "422" in str(caught.value)


async def test_a_plain_detail_is_kept_but_capped():
    long = httpx.Response(409, json={"detail": "seg_id m-1-x was already saved " + "x" * 500})

    with pytest.raises(BrainRejected) as caught:
        await client(Recorder(long)).ingest_segments("m-1", [seg("m-1", 1)])

    assert "already saved" in str(caught.value)
    assert len(str(caught.value)) < 300


async def test_nothing_to_send_makes_no_request():
    recorder = Recorder()

    await client(recorder).ingest_segments("m-1", [])

    assert recorder.requests == []


# invoke


def invocation() -> Invocation:
    return Invocation(
        id="inv-1",
        meeting_id="m-1",
        via="voice",
        visibility="public",
        asked_by_id="u-alex",
        asked_by_name="Alex Chen",
        question="what did we decide about Postgres?",
        t=12.0,
    )


async def test_invoke_posts_the_question_with_recent_segments_and_returns_the_answer():
    answer = Answer(id="a-1", invocation_id="inv-1", text="Keep Postgres (standup, 03:12).")
    recorder = Recorder(httpx.Response(200, json={"answer": answer.model_dump(mode="json")}))

    got = await client(recorder).invoke(invocation(), [seg("m-1", 1)])

    assert got == answer
    [request] = recorder.requests
    assert str(request.url) == "http://brain.test/internal/meetings/m-1/invoke"
    body = httpx.Response(200, content=request.content).json()
    assert body["invocation"]["id"] == "inv-1"
    assert [s["seg_id"] for s in body["recent_segments"]] == ["m-1-u-alex-1"]


@pytest.mark.parametrize("failure", [503, httpx.ReadTimeout("slow")])
async def test_invoke_is_never_retried_because_it_is_not_idempotent(failure):
    recorder = Recorder(failure, 200)

    with pytest.raises(BrainUnavailable):
        await client(recorder).invoke(invocation(), [])

    assert len(recorder.requests) == 1


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
    await store.add_participant(meeting.id, "u-alex")  # joined through the link
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


# translation (#106)


async def test_translate_posts_the_speech_and_the_detected_language():
    answer = TranslateResponse(language="es", text="We keep Postgres for now.")
    recorder = Recorder(json_response(answer))

    got = await client(recorder).translate("m-1", "Por ahora nos quedamos con Postgres.", "es")

    assert got == answer
    [request] = recorder.requests
    assert str(request.url) == "http://brain.test/internal/meetings/m-1/translate"
    assert request.headers["X-Internal-Token"] == TOKEN
    body = httpx.Response(200, content=request.content).json()
    assert body == {"text": "Por ahora nos quedamos con Postgres.", "language": "es"}


@pytest.mark.parametrize("failure", [503, 502, httpx.ReadTimeout("slow")])
async def test_translate_is_tried_once_because_a_late_caption_is_useless(failure):
    recorder = Recorder(failure, json_response(TranslateResponse(language="es", text="x")))

    with pytest.raises(BrainUnavailable):
        await client(recorder).translate("m-1", "Hola", None)

    assert len(recorder.requests) == 1


async def test_a_failed_translate_says_which_status_so_the_worker_can_back_off():
    with pytest.raises(BrainUnavailable) as off:
        await client(Recorder(503)).translate("m-1", "Hola", None)
    with pytest.raises(BrainUnavailable) as slow:
        await client(Recorder(httpx.ReadTimeout("slow"))).translate("m-1", "Hola", None)

    assert off.value.status == 503
    assert slow.value.status is None


async def test_translate_against_the_real_brain_endpoint():
    from brain.api.deps import get_settings, get_store, get_translation_llm_factory
    from brain.config import Settings as BrainSettings
    from brain.llm import MockLLM
    from brain.main import create_app
    from brain.store import InMemoryStore
    from brain.translation import Translation
    from contracts import Team

    store = InMemoryStore(teams=[Team(id="t-1", name="Checkout", member_ids=["u-alex"])], people=[])
    meeting = await store.create_meeting("t-1", "Standup", host_id="u-alex")
    llm = MockLLM(structured={Translation: Translation(language="es", english="Hello everyone")})
    app = create_app()
    app.dependency_overrides[get_store] = lambda: store
    app.dependency_overrides[get_translation_llm_factory] = lambda: lambda: llm
    app.dependency_overrides[get_settings] = lambda: BrainSettings(
        _env_file=None, brain_internal_token=TOKEN
    )
    brain = HttpBrainClient("http://brain.test", TOKEN, transport=httpx.ASGITransport(app=app))

    got = await brain.translate(meeting.id, "Hola a todos", None)

    assert got == TranslateResponse(language="es", text="Hello everyone")


# what the worker reads when it joins


def json_response(model) -> httpx.Response:
    return httpx.Response(200, json=model.model_dump(mode="json"))


def a_meeting(meeting_id: str = "m-1") -> Meeting:
    return Meeting(
        id=meeting_id,
        team_id="t-1",
        title="Standup",
        status="live",
        code="abc-defg-hij",
        host_id="u-alex",
        participant_ids=["u-alex"],
        started_at=datetime(2026, 10, 3, 17, 0, tzinfo=UTC),
    )


async def test_reads_the_meeting_and_the_teams_voice():
    info = WorkerMeetingResponse(meeting=a_meeting(), voice_id="v-team")
    recorder = Recorder(500, json_response(info))

    got = await client(recorder).meeting("m-1")

    assert got == info
    assert [r.method for r in recorder.requests] == ["GET", "GET"]  # a read is retried
    assert str(recorder.requests[0].url) == "http://brain.test/internal/meetings/m-1"
    assert recorder.requests[0].headers["X-Internal-Token"] == TOKEN


async def test_an_unknown_meeting_is_a_rejection():
    with pytest.raises(BrainRejected) as caught:
        await client(Recorder(404)).meeting("not-a-meeting")

    assert caught.value.status == 404


async def test_says_the_agent_joined_and_retries_since_the_brain_keeps_the_first_time():
    recorder = Recorder(503, 204)

    await client(recorder).agent_joined("m-1")

    assert [r.method for r in recorder.requests] == ["POST", "POST"]
    assert str(recorder.requests[0].url) == "http://brain.test/internal/meetings/m-1/agent-joined"
    assert recorder.requests[0].headers["X-Internal-Token"] == TOKEN


async def test_a_meeting_that_is_not_live_refuses_the_join():
    with pytest.raises(BrainRejected) as caught:
        await client(Recorder(409)).agent_joined("m-1")

    assert caught.value.status == 409


async def test_reads_the_keyterms_for_the_meetings_scribe_streams():
    recorder = Recorder(httpx.Response(200, json={"terms": ["Polaris", "DS-104"]}))

    assert await client(recorder).keyterms("m-1") == ["Polaris", "DS-104"]
    assert str(recorder.requests[0].url) == "http://brain.test/internal/meetings/m-1/keyterms"


# public chat


def chat_message(**overrides) -> ChatMessage:
    return ChatMessage(
        **{
            "id": "lk-stream-1",
            "meeting_id": "m-1",
            "sender_id": "u-alex",
            "sender_name": "Alex Chen",
            "is_agent": False,
            "text": "@Polaris what's blocking DS-104?",
            "ts": datetime(2026, 10, 3, 17, 5, tzinfo=UTC),
        }
        | overrides
    )


async def test_public_chat_is_posted_and_retried_since_the_brain_keeps_one_copy_per_id():
    recorder = Recorder(503, 204)

    await client(recorder).ingest_public_chat("m-1", chat_message())

    assert len(recorder.requests) == 2
    assert str(recorder.requests[0].url) == "http://brain.test/internal/meetings/m-1/chat"
    body = httpx.Response(200, content=recorder.requests[0].content).json()
    assert body["id"] == "lk-stream-1"


async def test_private_chat_is_never_sent():
    recorder = Recorder()
    private = chat_message(text="just between us", visibility="private", recipient_id="u-sarah")

    with pytest.raises(ValueError):
        await client(recorder).ingest_public_chat("m-1", private)
    assert recorder.requests == []


# ticks


def an_agenda() -> Agenda:
    return Agenda(meeting_id="m-1", items=[], generated_at=datetime(2026, 10, 3, tzinfo=UTC))


async def test_an_agenda_tick_posts_and_returns_the_agenda_and_nudges():
    response = AgendaTrackResponse(
        agenda=an_agenda(),
        nudges=[AgendaNudge(meeting_id="m-1", item_id="i-1", text="Refunds hasn't come up")],
    )
    recorder = Recorder(json_response(response))

    got = await client(recorder).track_agenda("m-1")

    assert got == response
    [request] = recorder.requests
    assert (request.method, str(request.url)) == (
        "POST",
        "http://brain.test/internal/meetings/m-1/agenda/track",
    )


@pytest.mark.parametrize("status", [502, 503])
async def test_an_agenda_tick_whose_model_failed_still_returns_what_to_publish(status):
    """The brain saved the rule nudges as sent, so the worker must publish them."""
    body = {
        "detail": "Gemini is not configured",
        "agenda": an_agenda().model_dump(mode="json"),
        "nudges": [AgendaNudge(meeting_id="m-1", item_id="i-1", text="Nudge").model_dump()],
    }

    got = await client(Recorder(httpx.Response(status, json=body))).track_agenda("m-1")

    assert [n.text for n in got.nudges] == ["Nudge"]


async def test_an_agenda_tick_is_not_retried_within_the_tick():
    recorder = Recorder(504, 200)

    with pytest.raises(BrainUnavailable):
        await client(recorder).track_agenda("m-1")
    assert len(recorder.requests) == 1


async def test_a_fact_check_tick_returns_the_checks():
    response = FactCheckResponse(
        checks=[
            FactCheck(
                id="f-1",
                claim="PR 41 shipped",
                speaker_name="Bob",
                verdict="contradicted",
                confidence=0.9,
                severity="high",
            )
        ]
    )
    recorder = Recorder(json_response(response))

    got = await client(recorder).fact_check("m-1")

    assert got == response
    assert str(recorder.requests[0].url) == "http://brain.test/internal/meetings/m-1/fact-check"


async def test_a_failed_fact_check_tick_is_a_rejection_or_unavailable():
    with pytest.raises(BrainRejected):
        await client(Recorder(409)).fact_check("m-1")
    with pytest.raises(BrainUnavailable):
        await client(Recorder(503)).fact_check("m-1")


async def test_a_catch_up_posts_the_participant_and_span_and_returns_what_to_send():
    from contracts import CatchUpResponse

    response = CatchUpResponse(text="Catching you up (00:00 to 07:00):", source_times=[90.0])
    recorder = Recorder(json_response(response))

    got = await client(recorder).catch_up("m-1", "u-sarah", 0, 420)

    assert got == response
    [request] = recorder.requests
    assert str(request.url) == "http://brain.test/internal/meetings/m-1/catch-up"
    assert request.headers["X-Internal-Token"] == TOKEN
    assert json.loads(request.content) == {"participant_id": "u-sarah", "since": 0, "until": 420}


async def test_a_catch_up_is_not_retried_and_a_refusal_is_a_rejection():
    recorder = Recorder(503)
    with pytest.raises(BrainUnavailable):
        await client(recorder).catch_up("m-1", "u-sarah", 0, 420)
    assert len(recorder.requests) == 1
    with pytest.raises(BrainRejected):
        await client(Recorder(409)).catch_up("m-1", "u-sarah", 0, 420)


# who may act on the shared answer card


async def test_asks_whether_a_participant_may_act_on_the_card():
    recorder = Recorder(500, httpx.Response(200, json={"allowed": True}))

    assert await client(recorder).card_permission("m-1", "u-sarah") is True

    assert [r.method for r in recorder.requests] == ["GET", "GET"]  # a read is retried
    request = recorder.requests[0]
    assert request.url.path == "/internal/meetings/m-1/card-permission"
    assert request.url.params["participant_id"] == "u-sarah"
    assert request.headers["X-Internal-Token"] == TOKEN


async def test_a_refused_participant_is_not_allowed():
    recorder = Recorder(httpx.Response(200, json={"allowed": False}))

    assert await client(recorder).card_permission("m-1", "u-priya") is False


async def test_the_participant_id_is_sent_as_a_query_value_not_spliced_into_the_path():
    recorder = Recorder(httpx.Response(200, json={"allowed": False}))

    await client(recorder).card_permission("m-1", "u-x&participant_id=u-alex")

    assert recorder.requests[0].url.params.get_list("participant_id") == [
        "u-x&participant_id=u-alex"
    ]


async def test_a_card_permission_failure_raises():
    with pytest.raises(BrainUnavailable):
        await client(Recorder(503, 503, 503)).card_permission("m-1", "u-sarah")
    with pytest.raises(BrainRejected):
        await client(Recorder(404)).card_permission("m-1", "u-sarah")


async def test_card_permission_matches_the_real_brain():
    from brain.api.deps import get_settings, get_store
    from brain.config import Settings as BrainSettings
    from brain.main import create_app
    from brain.store import InMemoryStore
    from contracts import Person, Team

    alex = Person(id="u-alex", name="Alex Chen", short="Alex", initials="AC")
    sarah = Person(id="u-sarah", name="Sarah Kim", short="Sarah", initials="SK")
    store = InMemoryStore(
        teams=[Team(id="t-1", name="Checkout", member_ids=["u-alex", "u-sarah"])],
        people=[alex, sarah],
    )
    meeting = await store.create_meeting("t-1", "Standup", host_id="u-alex")
    for person in (alex, sarah):
        await store.add_participant(meeting.id, person.id)
    settings = await store.settings("t-1")
    await store.save_settings(settings.model_copy(update={"who_can_allow": "host"}))
    app = create_app()
    app.dependency_overrides[get_store] = lambda: store
    app.dependency_overrides[get_settings] = lambda: BrainSettings(
        _env_file=None, brain_internal_token=TOKEN
    )
    brain = HttpBrainClient(
        "http://brain.test", TOKEN, transport=httpx.ASGITransport(app=app), backoff=0
    )

    assert await brain.card_permission(meeting.id, "u-alex") is True
    assert await brain.card_permission(meeting.id, "u-sarah") is False
