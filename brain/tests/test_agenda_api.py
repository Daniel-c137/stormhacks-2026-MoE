"""The lobby agenda over HTTP: edit the timeboxed list, rewrite a rough topic, suggest items."""

import re
from datetime import UTC, date, datetime, timedelta

import anyio
import pytest
from api_support import ALEX, OUTSIDER, SARAH, TEAM, create
from fastapi.testclient import TestClient

from brain.agent.agenda import AgendaRewrite, AgendaSuggestionDraft, SuggestedItem
from brain.api.deps import get_llm, get_settings
from brain.llm import FallbackLLM, LLMOutOfCapacity, MockLLM
from contracts import (
    Agenda,
    AgendaItem,
    Decision,
    DecisionRelation,
    Report,
    Risk,
    Source,
    TaskDraft,
)


def agenda(client: TestClient, meeting_id: str):
    return client.get(f"/meetings/{meeting_id}/agenda")


def put(client: TestClient, meeting_id: str, *items: dict):
    return client.put(f"/meetings/{meeting_id}/agenda", json={"items": list(items)})


def saved(client: TestClient, meeting_id: str) -> list[dict]:
    response = agenda(client, meeting_id)
    assert response.status_code == 200, response.text
    return response.json()["items"]


def use_llm(app, llm) -> None:
    app.dependency_overrides[get_llm] = lambda: llm


# reading and editing


def test_a_new_meeting_has_an_empty_agenda(client_as):
    meeting = create(client_as(ALEX))

    body = agenda(client_as(SARAH), meeting["id"]).json()

    assert body["meeting_id"] == meeting["id"]
    assert body["items"] == []


def test_new_items_get_ids_and_the_person_who_added_them_in_the_order_sent(client_as):
    alex = client_as(ALEX)
    meeting = create(alex)

    response = put(
        alex,
        meeting["id"],
        {"title": "Job queue: stay on Postgres or move to Redis", "minutes": 15},
        {"title": "  Refund the double-charged users  ", "minutes": 10},
        {"title": "Release notes"},
    )

    assert response.status_code == 200, response.text
    items = response.json()["items"]
    assert [(i["title"], i["minutes"]) for i in items] == [
        ("Job queue: stay on Postgres or move to Redis", 15),
        ("Refund the double-charged users", 10),
        ("Release notes", None),
    ]
    assert all(i["id"] for i in items) and len({i["id"] for i in items}) == 3
    assert {i["added_by"] for i in items} == {ALEX.id}
    assert {i["status"] for i in items} == {"pending"}
    assert response.json()["updated_at"] is not None
    assert saved(alex, meeting["id"]) == items


def test_a_teammate_reorders_renames_removes_and_adds_items(client_as):
    # client_as switches who the app's requests are made as, so each call names its person.
    meeting = create(client_as(ALEX))
    first, second, third = put(
        client_as(ALEX),
        meeting["id"],
        {"title": "One", "minutes": 5},
        {"title": "Two"},
        {"title": "Three"},
    ).json()["items"]

    response = put(
        client_as(SARAH),
        meeting["id"],
        {"id": third["id"], "title": "Three, renamed", "minutes": 20},
        {"title": "Four"},
        {"id": first["id"], "title": "One", "minutes": None},
    )

    assert response.status_code == 200, response.text
    items = saved(client_as(ALEX), meeting["id"])
    assert [i["title"] for i in items] == ["Three, renamed", "Four", "One"]
    assert items[0]["id"] == third["id"] and items[0]["minutes"] == 20
    assert items[2]["id"] == first["id"] and items[2]["minutes"] is None
    assert [i["added_by"] for i in items] == [ALEX.id, SARAH.id, ALEX.id]
    assert second["id"] not in {i["id"] for i in items}


def test_editing_keeps_an_items_status_sources_why_and_who_added_it(client_as, store):
    alex = client_as(ALEX)
    meeting = create(alex)
    source = Source(kind="jira_issue", label="DS-117", url="https://x.example/browse/DS-117")
    proposed = AgendaItem(
        id="a-1",
        title="Refunds",
        why="Still open since last week",
        sources=[source],
        status="covered",
        minutes=10,
        added_by=None,
    )
    anyio.run(
        store.save_agenda,
        Agenda(meeting_id=meeting["id"], items=[proposed], generated_at=datetime.now(UTC)),
    )

    put(alex, meeting["id"], {"id": "a-1", "title": "Refunds for DS-117", "minutes": 5})

    (item,) = saved(alex, meeting["id"])
    assert item["title"] == "Refunds for DS-117" and item["minutes"] == 5
    assert item["status"] == "covered"
    assert item["why"] == "Still open since last week"
    assert item["sources"] == [source.model_dump()]
    assert item["added_by"] is None


def test_an_empty_list_clears_the_agenda(client_as):
    alex = client_as(ALEX)
    meeting = create(alex)
    put(alex, meeting["id"], {"title": "One"})

    assert put(alex, meeting["id"]).status_code == 200

    assert saved(alex, meeting["id"]) == []


def test_an_id_that_is_not_on_the_agenda_is_added_as_a_new_item(client_as):
    alex = client_as(ALEX)
    meeting = create(alex)

    (item,) = put(alex, meeting["id"], {"id": "made-up", "title": "One"}).json()["items"]

    assert item["id"] != "made-up"
    assert item["added_by"] == ALEX.id


@pytest.mark.parametrize(
    "bad",
    [
        {"title": ""},
        {"title": "   "},
        {"title": "x" * 201},
        {"title": "Zero", "minutes": 0},
        {"title": "Negative", "minutes": -5},
        {"title": "Too long", "minutes": 241},
    ],
)
def test_invalid_items_are_rejected_and_nothing_is_saved(client_as, bad):
    alex = client_as(ALEX)
    meeting = create(alex)
    put(alex, meeting["id"], {"title": "Keep me"})

    response = put(alex, meeting["id"], {"title": "Fine", "minutes": 240}, bad)

    assert response.status_code == 422
    assert [i["title"] for i in saved(alex, meeting["id"])] == ["Keep me"]


def test_the_same_item_twice_is_rejected(client_as):
    alex = client_as(ALEX)
    meeting = create(alex)
    (item,) = put(alex, meeting["id"], {"title": "One"}).json()["items"]

    response = put(
        alex, meeting["id"], {"id": item["id"], "title": "One"}, {"id": item["id"], "title": "1"}
    )

    assert response.status_code == 422


def test_a_scheduled_meeting_can_have_its_agenda_set_in_the_lobby(client_as, store):
    start = datetime.now(UTC) + timedelta(days=1)
    meeting = anyio.run(
        lambda: store.create_meeting(TEAM.id, "Planning", ALEX.id, scheduled_start=start)
    )
    assert meeting.status == "scheduled"

    response = put(client_as(SARAH), meeting.id, {"title": "Roadmap", "minutes": 30})

    assert response.status_code == 200, response.text


def test_the_agenda_is_read_only_once_the_meeting_has_ended(client_as):
    alex = client_as(ALEX)
    meeting = create(alex)
    put(alex, meeting["id"], {"title": "One"})
    alex.post(f"/meetings/{meeting['id']}/end")

    response = put(client_as(SARAH), meeting["id"], {"title": "Too late"})

    assert response.status_code == 409
    assert [i["title"] for i in saved(client_as(ALEX), meeting["id"])] == ["One"]


def test_another_team_can_neither_read_nor_edit_the_agenda(client_as):
    alex = client_as(ALEX)
    meeting = create(alex)
    put(alex, meeting["id"], {"title": "Private to the team"})
    olga = client_as(OUTSIDER)

    assert agenda(olga, meeting["id"]).status_code == 404
    assert put(olga, meeting["id"], {"title": "Hijack"}).status_code == 404
    assert [i["title"] for i in saved(client_as(ALEX), meeting["id"])] == ["Private to the team"]


def test_an_unknown_meeting_has_no_agenda(client_as):
    assert agenda(client_as(ALEX), "no-such-meeting").status_code == 404


# rewriting a rough topic


def test_a_rough_topic_comes_back_as_one_clear_item_and_nothing_is_saved(app, client_as):
    llm = MockLLM(
        structured={
            AgendaRewrite: AgendaRewrite(title="Job queue: stay on Postgres or move to Redis")
        }
    )
    use_llm(app, llm)
    alex = client_as(ALEX)
    meeting = create(alex)

    response = alex.post("/agenda/rewrite", json={"text": "redis vs postgres?"})

    assert response.status_code == 200, response.text
    assert response.json() == {"text": "Job queue: stay on Postgres or move to Redis"}
    (call,) = llm.calls
    assert "redis vs postgres?" in call.prompt
    assert call.system and "never add" in call.system.lower()
    assert saved(alex, meeting["id"]) == []


def test_a_rewrite_is_trimmed_to_a_single_line(app, client_as):
    messy = '  "Job queue:\n stay on Postgres   or move to Redis."  '
    use_llm(app, MockLLM(structured={AgendaRewrite: AgendaRewrite(title=messy)}))

    response = client_as(ALEX).post("/agenda/rewrite", json={"text": "redis vs postgres?"})

    assert response.json() == {"text": "Job queue: stay on Postgres or move to Redis"}


@pytest.mark.parametrize("text", ["", "   ", "x" * 1001])
def test_a_blank_or_huge_topic_is_rejected_without_asking_the_model(app, client_as, text):
    llm = MockLLM(structured={AgendaRewrite: AgendaRewrite(title="unused")})
    use_llm(app, llm)

    response = client_as(ALEX).post("/agenda/rewrite", json={"text": text})

    assert response.status_code == 422
    assert llm.calls == []


def test_a_failing_model_is_reported_not_papered_over(app, client_as):
    use_llm(app, MockLLM())

    response = client_as(ALEX).post("/agenda/rewrite", json={"text": "rate limit"})

    assert response.status_code == 502


class OutOfCapacity:
    """A provider whose every model is rate-limited."""

    last_model = None

    async def generate_structured(self, prompt, schema, *, system=None):
        raise LLMOutOfCapacity("every model tried: m: 429 RESOURCE_EXHAUSTED")


def test_the_rewrite_falls_back_to_the_next_provider_when_one_is_out_of_capacity(app, client_as):
    spare = MockLLM(structured={AgendaRewrite: AgendaRewrite(title="Rate limits")})
    use_llm(app, FallbackLLM([OutOfCapacity(), spare]))

    response = client_as(ALEX).post("/agenda/rewrite", json={"text": "rate limit"})

    assert response.status_code == 200, response.text
    assert response.json() == {"text": "Rate limits"}


def test_every_provider_out_of_capacity_is_a_502(app, client_as):
    use_llm(app, FallbackLLM([OutOfCapacity(), OutOfCapacity()]))

    response = client_as(ALEX).post("/agenda/rewrite", json={"text": "rate limit"})

    assert response.status_code == 502


def test_rewrite_and_suggest_are_unavailable_without_gemini(client_as):
    alex = client_as(ALEX)
    meeting = create(alex)

    rewrite = alex.post("/agenda/rewrite", json={"text": "rate limit"})
    suggest = alex.post(f"/meetings/{meeting['id']}/agenda/suggest")

    for response in (rewrite, suggest):
        assert response.status_code == 503
        assert "GEMINI_API_KEY" in response.json()["detail"]


# suggestions from earlier meetings, tasks and Jira


TODAY = datetime.now(UTC).date()


def earlier_meeting(client: TestClient, store, title: str = "Checkout sync") -> dict:
    """An ended meeting of the caller's team with a saved report."""
    meeting = create(client, title)
    client.post(f"/meetings/{meeting['id']}/end")
    mid = meeting["id"]

    def decision(n: int, text: str, **changes) -> Decision:
        return Decision(
            id=f"{mid}-d{n}", meeting_id=mid, text=text, made_by="Alex", t=60.0 * n, quote=text
        ).model_copy(update=changes)

    def task(n: int, title: str, **changes) -> TaskDraft:
        return TaskDraft(id=f"{mid}-t{n}", meeting_id=mid, title=title, t=30.0 * n).model_copy(
            update=changes
        )

    report = Report(
        meeting_id=mid,
        summary="Refunds and the job queue.",
        open_questions=["Who owns the refund script?"],
        blockers=["Staging database is down"],
        risks=[Risk(text="Queue backlog could delay emails", severity="high")],
        decisions=[
            decision(1, "Keep jobs on Postgres", status="superseded"),
            decision(
                2,
                "Move jobs to Redis",
                relation=DecisionRelation(type="contradicts", decision_id=f"{mid}-d1"),
            ),
            decision(3, "Ship on Fridays only"),
        ],
        tasks=[
            task(1, "Write the refund script"),
            task(2, "Rotate the API keys", due=date(2020, 1, 1), key="DS-90", jira_status="todo"),
            task(3, "Update the changelog", jira_status="done", due=date(2020, 1, 1)),
            task(4, "Dropped idea", include=False),
            task(
                5,
                "Plan the offsite",
                due=TODAY + timedelta(days=30),
                key="DS-91",
                jira_status="todo",
            ),
        ],
    )
    anyio.run(store.save_report, report)
    return meeting


def label_of(prompt: str, needle: str) -> str:
    """The input id the prompt gives the line that mentions `needle`."""
    match = re.search(rf"^\[(\w+)\][^\n]*{re.escape(needle)}", prompt, re.M | re.I)
    assert match, f"{needle!r} is not an input in the prompt:\n{prompt}"
    return match.group(1)


def suggest(client: TestClient, meeting_id: str):
    return client.post(f"/meetings/{meeting_id}/agenda/suggest")


def test_suggestions_come_from_earlier_reports_and_tasks_and_cite_them(app, client_as, store):
    alex = client_as(ALEX)
    earlier = earlier_meeting(alex, store)
    meeting = create(alex, "Next sync")

    def answer(prompt: str) -> AgendaSuggestionDraft:
        return AgendaSuggestionDraft(
            items=[
                SuggestedItem(
                    title="Who owns the refund script",
                    why="Left open last time",
                    minutes=10,
                    source_ids=[
                        label_of(prompt, "Who owns the refund script?"),
                        label_of(prompt, "Write the refund script"),
                    ],
                ),
                SuggestedItem(
                    title="Job queue: Postgres or Redis",
                    why="The decision was reversed",
                    minutes=500,
                    source_ids=[label_of(prompt, "Keep jobs on Postgres"), "made-up"],
                ),
                SuggestedItem(title="Invented topic", why="No basis", source_ids=["zzz"]),
                SuggestedItem(title="No sources at all", why="Hmm", source_ids=[]),
            ]
        )

    llm = MockLLM(structured={AgendaSuggestionDraft: answer})
    use_llm(app, llm)

    response = suggest(client_as(SARAH), meeting["id"])

    assert response.status_code == 200, response.text
    items = response.json()["items"]
    assert [i["title"] for i in items] == [
        "Who owns the refund script",
        "Job queue: Postgres or Redis",
    ]
    refund, queue = items
    assert refund["why"] == "Left open last time" and refund["minutes"] == 10
    assert queue["minutes"] is None
    for item in items:
        assert item["id"] and item["added_by"] is None and item["status"] == "pending"
        assert item["sources"]
        for source in item["sources"]:
            assert source["kind"] == "meeting" and source["meeting_id"] == earlier["id"]
            assert "Checkout sync" in source["label"]
    assert len(refund["sources"]) == 2
    assert queue["sources"][0]["t"] == 60.0
    assert saved(alex, meeting["id"]) == []


def test_only_open_and_overdue_work_and_flagged_decisions_are_offered(app, client_as, store):
    earlier_meeting(client_as(OUTSIDER), store, "Other team's sync")
    alex = client_as(ALEX)
    earlier_meeting(alex, store)
    meeting = create(alex)
    llm = MockLLM(structured={AgendaSuggestionDraft: AgendaSuggestionDraft(items=[])})
    use_llm(app, llm)

    suggest(alex, meeting["id"])

    (call,) = llm.calls
    for offered in [
        "Who owns the refund script?",
        "Staging database is down",
        "Queue backlog could delay emails",
        "Keep jobs on Postgres",
        "Move jobs to Redis",
        "Write the refund script",
        "Rotate the API keys",
    ]:
        label_of(call.prompt, offered)
    for withheld in [
        "Ship on Fridays only",
        "Update the changelog",
        "Dropped idea",
        "Plan the offsite",
        "Other team's sync",
    ]:
        assert withheld not in call.prompt


def test_at_most_six_suggestions(app, client_as, store):
    alex = client_as(ALEX)
    earlier_meeting(alex, store)
    meeting = create(alex)

    def answer(prompt: str) -> AgendaSuggestionDraft:
        source = label_of(prompt, "Staging database is down")
        return AgendaSuggestionDraft(
            items=[
                SuggestedItem(title=f"Item {n}", why="w", source_ids=[source]) for n in range(10)
            ]
        )

    use_llm(app, MockLLM(structured={AgendaSuggestionDraft: answer}))

    items = suggest(alex, meeting["id"]).json()["items"]

    assert [i["title"] for i in items] == [f"Item {n}" for n in range(6)]


def test_with_nothing_to_go_on_there_are_no_suggestions_and_no_model_call(app, client_as):
    alex = client_as(ALEX)
    meeting = create(alex)
    llm = MockLLM(structured={AgendaSuggestionDraft: AgendaSuggestionDraft(items=[])})
    use_llm(app, llm)

    response = suggest(alex, meeting["id"])

    assert response.status_code == 200
    assert response.json()["items"] == []
    assert llm.calls == []


def test_jira_is_listed_as_unavailable_when_it_is_not_configured(app, client_as, store):
    alex = client_as(ALEX)
    earlier_meeting(alex, store)
    meeting = create(alex)
    llm = MockLLM(structured={AgendaSuggestionDraft: AgendaSuggestionDraft(items=[])})
    use_llm(app, llm)

    body = suggest(alex, meeting["id"]).json()

    assert any("Jira" in reason and "JIRA_MCP_URL" in reason for reason in body["unavailable"])
    assert "jira_issue" not in llm.calls[0].prompt


def test_another_team_cannot_ask_for_suggestions(app, client_as):
    meeting = create(client_as(ALEX))
    use_llm(app, MockLLM(structured={AgendaSuggestionDraft: AgendaSuggestionDraft(items=[])}))

    assert suggest(client_as(OUTSIDER), meeting["id"]).status_code == 404


def jira_issue(
    key: str, summary: str, status: str = "In Progress", category: str = "indeterminate"
):
    return {
        "key": key,
        "fields": {
            "summary": summary,
            "status": {"name": status, "statusCategory": {"key": category}},
            "assignee": {"displayName": "Sarah Kim"},
        },
    }


@pytest.fixture
def jira_settings(app, settings, jira_over_http):
    jira, url = jira_over_http
    configured = settings.model_copy(
        update={
            "jira_mcp_url": url,
            "jira_project_key": "DS",
            "jira_base_url": "https://dropsubs.atlassian.net",
        }
    )
    app.dependency_overrides[get_settings] = lambda: configured
    return jira


def test_unfinished_jira_work_is_read_and_cited_with_a_link(app, client_as, jira_settings):
    jira = jira_settings
    jira.issues = [
        jira_issue("DS-117", "Refund the double-charged users"),
        jira_issue("DS-104", "Old cleanup", status="Done", category="done"),
    ]
    alex = client_as(ALEX)
    meeting = create(alex)

    def answer(prompt: str) -> AgendaSuggestionDraft:
        assert "Old cleanup" not in prompt
        return AgendaSuggestionDraft(
            items=[
                SuggestedItem(
                    title="Refunds (DS-117)",
                    why="Still in progress in Jira",
                    source_ids=[label_of(prompt, "DS-117")],
                )
            ]
        )

    use_llm(app, MockLLM(structured={AgendaSuggestionDraft: answer}))

    body = suggest(alex, meeting["id"]).json()

    assert not any("Jira" in r for r in body["unavailable"])
    (item,) = body["items"]
    assert item["sources"] == [
        {
            "kind": "jira_issue",
            "label": "DS-117",
            "url": "https://dropsubs.atlassian.net/browse/DS-117",
            "meeting_id": None,
            "t": None,
        }
    ]
    (search,) = jira.searches
    assert "DS" in search["jql"] and "done" in search["jql"].lower()


def test_a_failing_jira_search_is_reported_and_the_rest_still_works(
    app, client_as, store, jira_settings
):
    jira_settings.search_error = "Jira is having a bad day"
    alex = client_as(ALEX)
    earlier_meeting(alex, store)
    meeting = create(alex)

    def answer(prompt: str) -> AgendaSuggestionDraft:
        return AgendaSuggestionDraft(
            items=[
                SuggestedItem(
                    title="Staging",
                    why="Blocked",
                    source_ids=[label_of(prompt, "Staging database is down")],
                )
            ]
        )

    use_llm(app, MockLLM(structured={AgendaSuggestionDraft: answer}))

    body = suggest(alex, meeting["id"]).json()

    assert [i["title"] for i in body["items"]] == ["Staging"]
    assert any("Jira" in r and "bad day" in r for r in body["unavailable"])


def test_a_failing_model_while_suggesting_is_reported(app, client_as, store):
    alex = client_as(ALEX)
    earlier_meeting(alex, store)
    meeting = create(alex)
    use_llm(app, MockLLM())

    assert suggest(alex, meeting["id"]).status_code == 502


@pytest.mark.parametrize(
    ("zone", "today"), [("America/Vancouver", "2026-10-03"), ("UTC", "2026-10-04")]
)
def test_suggestions_are_asked_for_on_the_teams_today(
    app, client_as, store, monkeypatch, zone, today
):
    # 02:00 UTC on 4 October is the evening of 3 October in Vancouver.
    monkeypatch.setattr("brain.zones.now", lambda: datetime(2026, 10, 4, 2, tzinfo=UTC))
    alex = client_as(ALEX)
    team_settings = alex.get("/settings").json() | {"timezone": zone}
    assert alex.put("/settings", json=team_settings).status_code == 200
    earlier_meeting(alex, store)
    meeting = create(alex, "Next sync")
    llm = MockLLM(structured={AgendaSuggestionDraft: AgendaSuggestionDraft(items=[])})
    use_llm(app, llm)

    assert suggest(alex, meeting["id"]).status_code == 200

    [call] = llm.calls
    assert f"Today: {today}" in call.prompt
