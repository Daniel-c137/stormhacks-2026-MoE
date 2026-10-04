"""The worker's fact-check tick over HTTP: internal token, live meetings only, one tick per
meeting at a time, and a model failure that leaves nothing checked."""

import asyncio
from datetime import UTC, datetime

import anyio
import httpx
import pytest
from api_support import ALEX, SARAH, TEAM, WORKER_TOKEN
from conftest import FakeJira
from fact_check_support import (
    CONNECTED,
    CONTRADICTED,
    RELEASED,
    live,
    merged_after_release,
    say,
    scripted,
)
from fastapi.testclient import TestClient
from pipeline_support import GatedLLM

from brain.agent.factcheck import FactChecker, FactCheckPlan
from brain.api.deps import get_fact_checker
from brain.config import Settings
from brain.llm import MockLLM
from contracts import FactCheckResponse

pytestmark = pytest.mark.anyio


@pytest.fixture
def github(fake_github):
    merged_after_release(fake_github)
    return fake_github


@pytest.fixture
def use_llm(app, store, github):
    """use_llm(llm): the app's fact-checker answers with `llm` against the fakes."""
    jira = FakeJira(issue_reads=True)

    def use(llm) -> None:
        checker = FactChecker(
            llm,
            store,
            settings=Settings(_env_file=None, **CONNECTED),
            jira_target=jira.server,
            github_target=github.server,
        )
        app.dependency_overrides[get_fact_checker] = lambda: checker

    return use


def contradicting() -> MockLLM:
    return scripted(*CONTRADICTED)


def tick(client: TestClient, meeting_id: str, now: float | None = 60, **kwargs):
    body = {} if now is None else {"now": now}
    return client.post(f"/internal/meetings/{meeting_id}/fact-check", json=body, **kwargs)


def live_meeting(store, **team):
    return anyio.run(lambda: live(store, **team))


def test_a_tick_returns_checks_and_the_raised_hand(worker, store, use_llm):
    meeting = live_meeting(store)
    anyio.run(say, store, meeting, (SARAH, RELEASED, 10))
    use_llm(contradicting())

    response = tick(worker, meeting.id)

    assert response.status_code == 200, response.text
    body = FactCheckResponse.model_validate(response.json())
    [fact] = body.checks
    assert (fact.verdict, fact.raised_hand, fact.visibility) == ("contradicted", True, "public")
    assert (body.agent_state.state, body.agent_state.hand_urgency) == ("hand_raised", "critical")


def test_the_tick_needs_the_internal_token(app, store, use_llm):
    meeting = live_meeting(store)
    llm = contradicting()
    use_llm(llm)
    anonymous = TestClient(app)

    assert tick(anonymous, meeting.id).status_code == 401
    assert tick(anonymous, meeting.id, headers={"X-Internal-Token": "wrong"}).status_code == 401
    assert llm.calls == []


def test_only_a_live_meeting_is_fact_checked(worker, store, use_llm):
    llm = contradicting()
    use_llm(llm)
    later = datetime(2030, 1, 1, tzinfo=UTC)
    scheduled = anyio.run(
        lambda: store.create_meeting(TEAM.id, "Later", ALEX.id, scheduled_start=later)
    )
    ended = live_meeting(store)
    anyio.run(say, store, ended, (SARAH, RELEASED, 10))
    anyio.run(lambda: store.transition_status(ended.id, {"live"}, "processing"))

    assert tick(worker, "no-such-meeting").status_code == 404
    assert tick(worker, scheduled.id).status_code == 409
    assert tick(worker, ended.id).status_code == 409
    assert llm.calls == []


def test_now_defaults_to_the_time_since_the_meeting_started(worker, store, use_llm):
    meeting = live_meeting(store)
    use_llm(contradicting())

    response = tick(worker, meeting.id, now=None)

    assert response.status_code == 200, response.text
    assert tick(worker, meeting.id, now=-1).status_code == 422


def test_a_failed_model_call_is_a_502_and_the_stretch_is_checked_again(worker, store, use_llm):
    meeting = live_meeting(store)
    anyio.run(say, store, meeting, (SARAH, RELEASED, 10))
    use_llm(MockLLM())  # nothing scripted: every call fails

    response = tick(worker, meeting.id)

    assert response.status_code == 502
    assert anyio.run(store.fact_check_state, meeting.id) is None

    use_llm(contradicting())
    assert len(tick(worker, meeting.id).json()["checks"]) == 1


def test_quiet_needs_no_model_and_an_unconfigured_model_is_a_503(worker, store):
    meeting = live_meeting(store, sensitivity="quiet")
    anyio.run(say, store, meeting, (SARAH, RELEASED, 10))

    quiet = tick(worker, meeting.id)

    assert quiet.status_code == 200, quiet.text
    assert quiet.json()["checks"] == []

    live_meeting(store, sensitivity="balanced")  # the team turns fact-checks on
    unconfigured = tick(worker, meeting.id)
    assert unconfigured.status_code == 503
    assert "Gemini" in unconfigured.json()["detail"]


async def test_overlapping_ticks_for_one_meeting_check_a_stretch_once(app, store, use_llm):
    meeting = await live(store)
    await say(store, meeting, (SARAH, RELEASED, 10))
    llm = GatedLLM(structured=contradicting()._structured)
    use_llm(llm)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://brain",
        headers={"X-Internal-Token": WORKER_TOKEN},
    ) as http:
        url = f"/internal/meetings/{meeting.id}/fact-check"
        first = asyncio.ensure_future(http.post(url, json={"now": 60}))
        await llm.called.wait()
        second = asyncio.ensure_future(http.post(url, json={"now": 61}))
        await asyncio.sleep(0.05)
        llm.gate.set()
        responses = [await first, await second]

    assert [r.status_code for r in responses] == [200, 200]
    assert [len(r.json()["checks"]) for r in responses] == [1, 0]
    assert len([c for c in llm.calls if c.schema is FactCheckPlan]) == 1
    assert len(await store.fact_checks(meeting.id)) == 1
