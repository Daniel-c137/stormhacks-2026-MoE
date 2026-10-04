"""Keeping time against the agenda: the worker's timer tick classifies new final segments, gives
their time to the current item, marks covered items and returns visual nudges."""

from datetime import UTC, datetime, timedelta

import anyio
import httpx
import pytest
from api_support import ALEX, SARAH, TEAM, WORKER_TOKEN, create, postgres_world, speakers_join
from fastapi.testclient import TestClient

from brain.agent.timekeeping import MAX_WAIT_S, MIN_TALK_S, NOW_SLACK_S, SETTLE_S, AgendaTrackDraft
from brain.api.deps import current_user, get_llm_factory, get_settings, get_store
from brain.llm import LLMError, MockLLM
from brain.main import create_app
from brain.store import Conflict
from contracts import Agenda, AgendaItem, TranscriptSegment

HOUR = 3600.0


class Classifier:
    """A MockLLM that answers each classification with the next scripted reply, in order."""

    def __init__(self):
        self.replies: list[AgendaTrackDraft] = []
        self.llm = MockLLM(structured={AgendaTrackDraft: self._next})

    def says(self, current: str | None = None, covered: tuple[str, ...] = ()) -> "Classifier":
        self.replies.append(AgendaTrackDraft(current=current, covered=list(covered)))
        return self

    def _next(self, prompt: str) -> AgendaTrackDraft:
        if not self.replies:
            raise LLMError("no scripted classification left")
        return self.replies.pop(0)

    @property
    def prompts(self) -> list[str]:
        return [call.prompt for call in self.llm.calls]


@pytest.fixture
def model(app) -> Classifier:
    classifier = Classifier()
    app.dependency_overrides[get_llm_factory] = lambda: lambda: classifier.llm
    return classifier


def said(meeting_id: str, n: int, text: str, start: float, end: float, speaker=ALEX) -> dict:
    return {
        "seg_id": f"{meeting_id}-{n}",
        "meeting_id": meeting_id,
        "speaker_id": speaker.id,
        "speaker_name": speaker.name,
        "text": text,
        "is_final": True,
        "t_start": start,
        "t_end": end,
    }


def ingest(worker: TestClient, meeting_id: str, *segments: dict) -> None:
    anyio.run(speakers_join, worker.app, meeting_id, list(segments))
    response = worker.post(
        f"/internal/meetings/{meeting_id}/segments", json={"segments": list(segments)}
    )
    assert response.status_code == 204, response.text


def track(worker: TestClient, meeting_id: str, now: float | None = None) -> httpx.Response:
    body = {} if now is None else {"now": now}
    return worker.post(f"/internal/meetings/{meeting_id}/agenda/track", json=body)


def tracked(worker: TestClient, meeting_id: str, now: float | None = None) -> dict:
    response = track(worker, meeting_id, now)
    assert response.status_code == 200, response.text
    return response.json()


def plan(client: TestClient, meeting_id: str, *items: tuple[str, int | None]) -> list[str]:
    """Saves the agenda through the lobby's PUT; returns the item ids in order."""
    response = client.put(
        f"/meetings/{meeting_id}/agenda",
        json={"items": [{"title": title, "minutes": minutes} for title, minutes in items]},
    )
    assert response.status_code == 200, response.text
    return [item["id"] for item in response.json()["items"]]


STANDUP = (("Waitlist email", 10), ("Refund policy", 5), ("Launch date", 5))


def by_id(agenda: dict) -> dict[str, dict]:
    return {item["id"]: item for item in agenda["items"]}


def T() -> datetime:
    return datetime.now(UTC)


def seed(store, agenda: Agenda) -> None:
    anyio.run(store.save_agenda, agenda)


def live_meeting(store, *, duration_min: int | None = None, started_ago_s: float = HOUR):
    """A live meeting that started `started_ago_s` ago, so ticks may name times up to then."""
    meeting = anyio.run(
        lambda: store.create_meeting(TEAM.id, "Standup", ALEX.id, duration_min=duration_min)
    )
    started = datetime.now(UTC) - timedelta(seconds=started_ago_s)
    return anyio.run(store.update_meeting, meeting.model_copy(update={"started_at": started}))


def item(item_id: str, title: str, minutes: int | None, **state) -> AgendaItem:
    return AgendaItem(id=item_id, title=title, minutes=minutes, **state)


def standup(store, client_as) -> tuple[str, list[str]]:
    """A live meeting with the STANDUP agenda saved through the lobby: (meeting id, item ids)."""
    meeting = live_meeting(store).id
    return meeting, plan(client_as(ALEX), meeting, *STANDUP)


def saved_agenda(store, meeting_id: str) -> Agenda:
    agenda = anyio.run(store.agenda, meeting_id)
    assert agenda is not None
    return agenda


async def rename_first(store, meeting_id: str, title: str = "Waitlist email, renamed") -> None:
    """Another writer (a lobby edit) renames the first item and commits."""
    agenda = await store.agenda(meeting_id)
    items = [agenda.items[0].model_copy(update={"title": title}), *agenda.items[1:]]
    await store.save_agenda(agenda.model_copy(update={"items": items}))


# classifying the new stretch


def test_the_first_tick_gives_the_stretch_to_the_item_it_is_about(worker, client_as, store, model):
    meeting, (waitlist, refunds, launch) = standup(store, client_as)
    ingest(
        worker,
        meeting,
        said(meeting, 1, "The waitlist email draft is ready for review", 0, 8),
        said(meeting, 2, "I'd cut the second paragraph of the waitlist email", 9, 20, SARAH),
        said(meeting, 3, "Fine, I'll send it Thursday", 22, 30),
    )
    model.says(current="a1")

    body = tracked(worker, meeting, now=60)

    agenda = body["agenda"]
    assert agenda["current_item_id"] == waitlist
    assert agenda["tracked_until"] == 60 - SETTLE_S
    assert [i["discussed_s"] for i in agenda["items"]] == [30, 0, 0]
    assert {i["status"] for i in agenda["items"]} == {"pending"}
    assert agenda["revision"] == 2  # the lobby's save, then this tick's
    assert body["nudges"] == []
    [prompt] = model.prompts
    assert "Waitlist email" in prompt and "Refund policy" in prompt and "Launch date" in prompt
    assert "I'd cut the second paragraph of the waitlist email" in prompt
    assert "Sarah Kim" in prompt
    saved = client_as(ALEX).get(f"/meetings/{meeting}/agenda").json()
    assert saved == agenda
    assert {refunds, launch} <= set(by_id(saved))


def test_the_next_tick_sends_only_the_new_segments(worker, client_as, store, model):
    meeting, (waitlist, refunds, _) = standup(store, client_as)
    ingest(worker, meeting, said(meeting, 1, "Waitlist email goes out Thursday", 0, 10))
    model.says(current="a1").says(current="a2")
    tracked(worker, meeting, now=70)

    ingest(
        worker,
        meeting,
        said(meeting, 2, "Next, the refund policy for annual plans", 80, 90, SARAH),
        said(meeting, 3, "Prorated refunds within thirty days", 92, 100),
    )
    body = tracked(worker, meeting, now=130)

    first, second = model.prompts
    assert "Waitlist email goes out Thursday" in first
    assert "Waitlist email goes out Thursday" not in second
    assert "Next, the refund policy for annual plans" in second
    assert "Prorated refunds within thirty days" in second
    items = by_id(body["agenda"])
    assert body["agenda"]["current_item_id"] == refunds
    assert (items[waitlist]["discussed_s"], items[refunds]["discussed_s"]) == (10, 20)


def test_segments_that_just_ended_wait_for_the_next_tick(worker, client_as, store, model):
    meeting, _ = standup(store, client_as)
    ingest(
        worker,
        meeting,
        said(meeting, 1, "Settled words", 0, 20),
        said(meeting, 2, "Words still settling", 21, 29),
    )
    model.says(current="a1").says(current="a1")

    tracked(worker, meeting, now=30)
    tracked(worker, meeting, now=30 + MAX_WAIT_S)

    first, second = model.prompts
    assert "Settled words" in first and "Words still settling" not in first
    assert "Words still settling" in second and "Settled words" not in second


# how often the model is asked


def test_ticks_every_few_seconds_do_not_ask_the_model_per_utterance(
    worker, client_as, store, model
):
    meeting, (waitlist, *_) = standup(store, client_as)
    for _ in range(10):
        model.says(current="a1")

    for k in range(10):
        ingest(worker, meeting, said(meeting, k, f"Waitlist point {k}", 3 * k, 3 * k + 2))
        body = tracked(worker, meeting, now=3 * k + 3 + SETTLE_S)

    # Once: when the stretch first held MIN_TALK_S of talk, at the seventh tick.
    assert 3 * 6 + 2 >= MIN_TALK_S > 3 * 5 + 2
    [prompt] = model.prompts
    assert "Waitlist point 6" in prompt and "Waitlist point 7" not in prompt
    assert body["agenda"]["tracked_until"] == 3 * 6 + 3
    assert by_id(body["agenda"])[waitlist]["discussed_s"] == 20


def test_a_short_remark_is_classified_once_the_stretch_is_long_enough(
    worker, client_as, store, model
):
    meeting, (waitlist, *_) = standup(store, client_as)
    ingest(worker, meeting, said(meeting, 1, "Quick waitlist question", 10, 14))
    model.says(current="a1")

    waiting = tracked(worker, meeting, now=MAX_WAIT_S)
    due = tracked(worker, meeting, now=MAX_WAIT_S + SETTLE_S)

    assert waiting["agenda"]["tracked_until"] is None
    assert len(model.prompts) == 1
    assert due["agenda"]["tracked_until"] == MAX_WAIT_S
    assert by_id(due["agenda"])[waitlist]["discussed_s"] == 4


# time attribution


def test_long_pauses_and_overlapping_speakers_are_not_counted_twice(
    worker, client_as, store, model
):
    meeting, (waitlist, *_) = standup(store, client_as)
    ingest(
        worker,
        meeting,
        said(meeting, 1, "Waitlist copy", 0, 10),
        said(meeting, 2, "Agreed, waitlist copy", 5, 12, SARAH),  # overlaps Alex
        said(meeting, 3, "Anything else on the waitlist?", 60, 70),  # after a long pause
    )
    model.says(current="a1")

    body = tracked(worker, meeting, now=100)

    assert by_id(body["agenda"])[waitlist]["discussed_s"] == 22


def test_an_utterance_across_a_tick_counts_on_both_sides_of_it(worker, client_as, store, model):
    meeting, (waitlist, *_) = standup(store, client_as)
    ingest(
        worker,
        meeting,
        said(meeting, 1, "Waitlist email first half", 0, 20),
        said(meeting, 2, "and the long second half", 25, 45),  # still going at the first tick
    )
    model.says(current="a1").says(current="a1")

    first = tracked(worker, meeting, now=35)
    second = tracked(worker, meeting, now=30 + MAX_WAIT_S + SETTLE_S)

    assert by_id(first["agenda"])[waitlist]["discussed_s"] == 30
    assert by_id(second["agenda"])[waitlist]["discussed_s"] == 45
    assert "and the long second half" in model.prompts[1]


def test_a_pause_across_a_tick_counts_like_any_other_pause(worker, client_as, store, model):
    meeting, (waitlist, *_) = standup(store, client_as)
    ingest(worker, meeting, said(meeting, 1, "Waitlist email", 0, 20))
    model.says(current="a1").says(current="a1")
    tracked(worker, meeting, now=30)

    ingest(worker, meeting, said(meeting, 2, "One more waitlist thing", 33, 60))  # 13 s pause
    body = tracked(worker, meeting, now=70)

    assert by_id(body["agenda"])[waitlist]["discussed_s"] == 60


def test_an_off_agenda_stretch_counts_for_no_item_and_clears_the_current_one(
    worker, client_as, store, model
):
    meeting, _ = standup(store, client_as)
    ingest(worker, meeting, said(meeting, 1, "Waitlist email", 0, 25))
    model.says(current="a1").says(current=None)
    tracked(worker, meeting, now=30)

    ingest(worker, meeting, said(meeting, 2, "Did anyone watch the game?", 30, 55, SARAH))
    body = tracked(worker, meeting, now=65)

    assert body["agenda"]["current_item_id"] is None
    assert [i["discussed_s"] for i in body["agenda"]["items"]] == [25, 0, 0]


# covered items and ids the model made up


def test_items_the_team_finished_are_marked_covered_and_stay_covered(
    worker, client_as, store, model
):
    meeting, (waitlist, refunds, launch) = standup(store, client_as)
    ingest(worker, meeting, said(meeting, 1, "Waitlist email is done, next topic", 0, 25))
    model.says(current="a1", covered=("a1",)).says(current="a2")
    tracked(worker, meeting, now=30)

    ingest(worker, meeting, said(meeting, 2, "Refund policy for annual plans", 30, 55))
    body = tracked(worker, meeting, now=65)

    statuses = {i["id"]: i["status"] for i in body["agenda"]["items"]}
    assert statuses == {waitlist: "covered", refunds: "pending", launch: "pending"}


def test_ids_that_are_not_on_the_agenda_are_ignored(worker, client_as, store, model):
    meeting, _ = standup(store, client_as)
    ingest(worker, meeting, said(meeting, 1, "Something", 0, 25))
    model.says(current="a9", covered=("a7", "Waitlist email", ""))

    body = tracked(worker, meeting, now=30)

    agenda = body["agenda"]
    assert agenda["current_item_id"] is None
    assert [(i["status"], i["discussed_s"]) for i in agenda["items"]] == [("pending", 0)] * 3
    assert agenda["tracked_until"] == 30 - SETTLE_S


# when the model is not asked


def test_nothing_new_since_the_last_tick_asks_the_model_nothing(worker, client_as, store, model):
    meeting, _ = standup(store, client_as)
    ingest(worker, meeting, said(meeting, 1, "Waitlist email", 0, 25))
    model.says(current="a1")
    first = tracked(worker, meeting, now=30)

    again = tracked(worker, meeting, now=30)
    later = tracked(worker, meeting, now=60)

    assert len(model.prompts) == 1
    assert again["agenda"] == first["agenda"]
    assert later["agenda"]["tracked_until"] == 60 - SETTLE_S
    assert later["agenda"]["items"] == first["agenda"]["items"]


def test_without_agenda_items_the_model_is_not_asked(worker, client_as, store, model):
    alex = client_as(ALEX)
    meeting = live_meeting(store).id
    ingest(worker, meeting, said(meeting, 1, "No agenda today", 0, 30))

    unplanned = tracked(worker, meeting, now=60)
    plan(alex, meeting)  # an empty list, saved
    emptied = tracked(worker, meeting, now=60)

    assert model.prompts == []
    assert (unplanned["agenda"]["items"], unplanned["nudges"]) == ([], [])
    assert unplanned["agenda"]["meeting_id"] == meeting
    assert (emptied["agenda"]["items"], emptied["nudges"]) == ([], [])


def test_now_defaults_to_the_time_since_the_meeting_started(worker, store, model):
    meeting = live_meeting(store, started_ago_s=120)
    seed(store, Agenda(meeting_id=meeting.id, items=[item("w", "Waitlist", 10)], generated_at=T()))
    ingest(
        worker,
        meeting.id,
        said(meeting.id, 1, "Waitlist email", 10, 20),
        said(meeting.id, 2, "Not yet said", 500, 510),
    )
    model.says(current="a1")

    response = worker.post(f"/internal/meetings/{meeting.id}/agenda/track")

    assert response.status_code == 200, response.text
    agenda = response.json()["agenda"]
    assert 120 - SETTLE_S <= agenda["tracked_until"] < 180
    assert agenda["items"][0]["discussed_s"] == 10
    assert "Not yet said" not in model.prompts[0]


# a bad clock


def test_now_beyond_the_real_time_since_the_start_is_rejected(worker, store, model):
    meeting = live_meeting(store, started_ago_s=120).id
    seed(store, Agenda(meeting_id=meeting, items=[item("w", "Waitlist", 10)], generated_at=T()))
    ingest(worker, meeting, said(meeting, 1, "Waitlist email", 10, 40))

    too_far = track(worker, meeting, now=120 + NOW_SLACK_S + 30)
    milliseconds = track(worker, meeting, now=120_000)

    assert (too_far.status_code, milliseconds.status_code) == (422, 422)
    assert saved_agenda(store, meeting).tracked_until is None
    assert model.prompts == []
    model.says(current="a1")
    assert track(worker, meeting, now=120 + NOW_SLACK_S - 5).status_code == 200


@pytest.mark.parametrize("raw", ["Infinity", "NaN", "-1"])
def test_a_now_that_is_not_a_finite_time_is_rejected(worker, store, model, raw):
    meeting = live_meeting(store).id

    response = worker.post(
        f"/internal/meetings/{meeting}/agenda/track",
        content=f'{{"now": {raw}}}',
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 422


# nudges


def waitlist_running_over(store, meeting_id: str) -> None:
    """Waitlist email (1 min) has had 55 s and is current; refunds and launch have not come up."""
    seed(
        store,
        Agenda(
            meeting_id=meeting_id,
            items=[
                item("w", "Waitlist email", 1, discussed_s=55),
                item("r", "Refund policy", 5),
                item("l", "Launch date", 5),
            ],
            generated_at=T(),
            current_item_id="w",
            tracked_until=0,
        ),
    )


def test_an_item_past_its_timebox_nudges_about_the_next_timeboxed_item_once(worker, store, model):
    meeting = live_meeting(store).id
    waitlist_running_over(store, meeting)
    ingest(worker, meeting, said(meeting, 1, "One more thing on the waitlist", 0, 20))
    model.says(current="a1").says(current="a1")

    body = tracked(worker, meeting, now=25)

    [nudge] = body["nudges"]
    assert nudge["meeting_id"] == meeting and nudge["item_id"] == "r"
    assert "Refund policy hasn't come up yet" in nudge["text"]
    assert "Waitlist email" in nudge["text"]
    assert by_id(body["agenda"])["r"]["nudged_t"] == 25
    assert by_id(body["agenda"])["l"]["nudged_t"] is None

    ingest(worker, meeting, said(meeting, 2, "Still the waitlist", 20, 40))
    again = tracked(worker, meeting, now=50)

    assert again["nudges"] == []
    assert by_id(again["agenda"])["w"]["discussed_s"] == 95


def test_no_timebox_nudge_while_the_item_is_within_its_timebox(worker, store, model):
    meeting = live_meeting(store).id
    waitlist_running_over(store, meeting)
    ingest(worker, meeting, said(meeting, 1, "Waitlist", 0, 4))
    model.says(current="a1")

    body = tracked(worker, meeting, now=MAX_WAIT_S + SETTLE_S)

    assert by_id(body["agenda"])["w"]["discussed_s"] == 59
    assert body["nudges"] == []


def test_no_timebox_nudge_when_no_timeboxed_item_is_still_pending(worker, store, model):
    meeting = live_meeting(store).id
    seed(
        store,
        Agenda(
            meeting_id=meeting,
            items=[
                item("w", "Waitlist email", 1, discussed_s=55),
                item("r", "Refund policy", 5, status="covered", discussed_s=120),
                item("l", "Launch date", None),
            ],
            generated_at=T(),
            current_item_id="w",
            tracked_until=0,
        ),
    )
    ingest(worker, meeting, said(meeting, 1, "Waitlist", 0, 30))
    model.says(current="a1")

    body = tracked(worker, meeting, now=40)

    assert by_id(body["agenda"])["w"]["discussed_s"] == 85
    assert body["nudges"] == []


def near_the_end_agenda(meeting_id: str) -> Agenda:
    return Agenda(
        meeting_id=meeting_id,
        items=[
            item("w", "Waitlist email", 10, discussed_s=600),
            item("r", "Refund policy", 5),
            item("d", "Docs", None, status="covered"),
            item("l", "Launch date", None),
        ],
        generated_at=T(),
        current_item_id="w",
        tracked_until=0,
    )


def test_near_the_scheduled_end_each_item_that_has_not_come_up_is_nudged_once(worker, store, model):
    meeting = live_meeting(store, duration_min=30).id
    seed(store, near_the_end_agenda(meeting))

    early = tracked(worker, meeting, now=24 * 60)
    near = tracked(worker, meeting, now=26 * 60)
    later = tracked(worker, meeting, now=28 * 60)

    assert early["nudges"] == []
    assert [(n["item_id"], n["text"]) for n in near["nudges"]] == [
        ("r", "Refund policy hasn't come up yet; 4 min left."),
        ("l", "Launch date hasn't come up yet; 4 min left."),
    ]
    assert later["nudges"] == []
    assert by_id(later["agenda"])["r"]["nudged_t"] == 26 * 60
    assert model.prompts == []  # time passing alone needs no model


def scheduled_then_started(store, *, late: timedelta, ago: timedelta) -> str:
    """A 30 min meeting scheduled `late` before it actually started, `ago` before now."""
    started = datetime.now(UTC) - ago
    meeting = anyio.run(
        lambda: store.create_meeting(
            TEAM.id, "Planning", ALEX.id, scheduled_start=started - late, duration_min=30
        )
    )
    anyio.run(store.start_meeting, meeting.id, started)
    seed(store, Agenda(meeting_id=meeting.id, items=[item("r", "Refunds", 5)], generated_at=T()))
    return meeting.id


def test_the_end_is_the_scheduled_one_when_a_meeting_starts_a_little_late(worker, store, model):
    meeting = scheduled_then_started(store, late=timedelta(minutes=10), ago=timedelta(minutes=20))

    body = tracked(worker, meeting, now=16 * 60)

    assert [n["text"] for n in body["nudges"]] == ["Refunds hasn't come up yet; 4 min left."]


@pytest.mark.parametrize("late", [timedelta(hours=2), timedelta(minutes=28)])
def test_a_meeting_started_after_its_slot_ran_out_gets_its_full_duration(
    worker, store, model, late
):
    meeting = scheduled_then_started(store, late=late, ago=timedelta(minutes=40))

    first = tracked(worker, meeting, now=60)
    near = tracked(worker, meeting, now=26 * 60)

    assert first["nudges"] == []
    assert [n["text"] for n in near["nudges"]] == ["Refunds hasn't come up yet; 4 min left."]


def test_without_a_duration_there_is_no_end_nudge(worker, store, model):
    meeting = live_meeting(store, started_ago_s=7 * HOUR).id
    seed(store, Agenda(meeting_id=meeting, items=[item("r", "Refunds", 5)], generated_at=T()))

    body = tracked(worker, meeting, now=6 * HOUR)

    assert body["nudges"] == []


# which meetings, and who may call


def test_only_a_live_meeting_is_tracked(worker, client_as, store, model):
    alex = client_as(ALEX)
    ended = create(alex)["id"]
    alex.post(f"/meetings/{ended}/end")
    scheduled = anyio.run(
        lambda: store.create_meeting(
            TEAM.id, "Later", ALEX.id, scheduled_start=datetime.now(UTC) + timedelta(days=1)
        )
    )

    assert track(worker, ended, now=0).status_code == 409
    assert track(worker, scheduled.id, now=0).status_code == 409
    assert track(worker, "no-such-meeting", now=0).status_code == 404
    assert model.prompts == []


def test_the_worker_token_is_required(app, worker, client_as, model):
    meeting = create(client_as(ALEX))["id"]

    anonymous = TestClient(app).post(f"/internal/meetings/{meeting}/agenda/track", json={})
    impostor = TestClient(app, headers={"X-Internal-Token": "guess"}).post(
        f"/internal/meetings/{meeting}/agenda/track", json={}
    )

    assert (anonymous.status_code, impostor.status_code) == (401, 401)


# failures


def test_a_failing_model_is_retried_on_the_next_tick(worker, client_as, store, model):
    meeting, (waitlist, *_) = standup(store, client_as)
    ingest(worker, meeting, said(meeting, 1, "Waitlist email", 0, 25))
    before = client_as(ALEX).get(f"/meetings/{meeting}/agenda").json()

    failed = track(worker, meeting, now=30)

    assert failed.status_code == 502
    assert (failed.json()["agenda"], failed.json()["nudges"]) == (before, [])
    assert client_as(ALEX).get(f"/meetings/{meeting}/agenda").json() == before

    model.says(current="a1")
    retried = tracked(worker, meeting, now=30)

    assert "Waitlist email" in model.prompts[-1]
    assert by_id(retried["agenda"])[waitlist]["discussed_s"] == 25


def test_a_failing_model_still_gets_the_rule_nudges_out(worker, store, model):
    meeting = live_meeting(store, duration_min=30).id
    seed(store, near_the_end_agenda(meeting))
    ingest(worker, meeting, said(meeting, 1, "Waitlist email, still", 26 * 60 - 40, 26 * 60 - 10))

    failed = track(worker, meeting, now=26 * 60)

    assert failed.status_code == 502
    body = failed.json()
    assert [n["item_id"] for n in body["nudges"]] == ["r", "l"]
    assert body["agenda"]["tracked_until"] == 0
    saved = saved_agenda(store, meeting)
    assert (saved.tracked_until, saved.items[1].nudged_t) == (0, 26 * 60)

    model.says(current="a1")
    retried = tracked(worker, meeting, now=26 * 60 + 10)

    assert retried["nudges"] == []
    assert by_id(retried["agenda"])["w"]["discussed_s"] == 630


def test_rule_nudges_need_no_gemini(worker, store):
    meeting = live_meeting(store, duration_min=30).id
    seed(store, near_the_end_agenda(meeting))

    body = tracked(worker, meeting, now=26 * 60)

    assert [n["item_id"] for n in body["nudges"]] == ["r", "l"]


def test_tracking_is_unavailable_without_gemini_only_when_the_model_is_needed(worker, store):
    meeting = live_meeting(store, duration_min=30).id
    seed(store, near_the_end_agenda(meeting))
    ingest(worker, meeting, said(meeting, 1, "Waitlist email, still", 26 * 60 - 40, 26 * 60 - 10))

    response = track(worker, meeting, now=26 * 60)

    assert response.status_code == 503
    assert [n["item_id"] for n in response.json()["nudges"]] == ["r", "l"]
    assert saved_agenda(store, meeting).tracked_until == 0


# human edits


def test_editing_the_agenda_keeps_the_tracking_state_of_items_that_remain(
    worker, client_as, store, model
):
    meeting, (waitlist, refunds, launch) = standup(store, client_as)
    ingest(worker, meeting, said(meeting, 1, "Waitlist email is done", 0, 25))
    model.says(current="a1", covered=("a1",))
    tracked(worker, meeting, now=30)

    response = client_as(SARAH).put(
        f"/meetings/{meeting}/agenda",
        json={
            "items": [
                {"id": refunds, "title": "Refunds for annual plans", "minutes": 10},
                {"id": waitlist, "title": "Waitlist email", "minutes": 15},
                {"title": "Pricing page"},
            ]
        },
    )

    assert response.status_code == 200, response.text
    agenda = response.json()
    assert (agenda["current_item_id"], agenda["tracked_until"]) == (waitlist, 30 - SETTLE_S)
    items = by_id(agenda)
    assert (items[waitlist]["status"], items[waitlist]["discussed_s"]) == ("covered", 25)
    assert items[waitlist]["minutes"] == 15
    assert (items[refunds]["status"], items[refunds]["discussed_s"]) == ("pending", 0)
    assert launch not in items

    response = client_as(SARAH).put(
        f"/meetings/{meeting}/agenda", json={"items": [{"id": refunds, "title": "Refunds"}]}
    )

    assert response.json()["current_item_id"] is None
    assert response.json()["tracked_until"] == 30 - SETTLE_S


def test_a_person_can_undo_a_wrong_covered_and_skip_an_item(worker, client_as, store, model):
    meeting, (waitlist, refunds, launch) = standup(store, client_as)
    ingest(worker, meeting, said(meeting, 1, "Everything is done", 0, 25))
    model.says(current="a1", covered=("a1", "a2", "a3"))
    tracked(worker, meeting, now=30)

    response = client_as(SARAH).put(
        f"/meetings/{meeting}/agenda",
        json={
            "items": [
                {"id": waitlist, "title": "Waitlist email"},  # no status: stays covered
                {"id": refunds, "title": "Refund policy", "status": "pending"},
                {"id": launch, "title": "Launch date", "status": "skipped"},
                {"title": "Pricing page", "status": "covered"},
            ]
        },
    )

    assert response.status_code == 200, response.text
    statuses = [i["status"] for i in response.json()["items"]]
    assert statuses == ["covered", "pending", "skipped", "covered"]
    bad = client_as(SARAH).put(
        f"/meetings/{meeting}/agenda",
        json={"items": [{"id": waitlist, "title": "Waitlist email", "status": "done"}]},
    )
    assert bad.status_code == 422


class Interfering:
    """The store, with another writer committing just before each of the next `times`
    conditional saves, as if it won the race."""

    def __init__(self, inner, before_save, times: int = 1):
        self.inner = inner
        self.before_save = before_save
        self.times = times
        self.conflicts = 0

    def __getattr__(self, name):
        return getattr(self.inner, name)

    async def save_agenda_if(self, agenda):
        if self.times:
            self.times -= 1
            await self.before_save(self.inner)
        try:
            return await self.inner.save_agenda_if(agenda)
        except Conflict:
            self.conflicts += 1
            raise


def test_an_edit_while_the_model_is_thinking_is_kept(app, worker, client_as, store):
    meeting, _ = standup(store, client_as)
    ingest(worker, meeting, said(meeting, 1, "Waitlist email", 0, 25))

    class EditingLLM(MockLLM):
        async def generate_structured(self, prompt, schema, *, system=None):
            await rename_first(store, meeting)
            return await super().generate_structured(prompt, schema, system=system)

    llm = EditingLLM(structured={AgendaTrackDraft: AgendaTrackDraft(current="a1")})
    app.dependency_overrides[get_llm_factory] = lambda: lambda: llm

    body = tracked(worker, meeting, now=30)

    first = body["agenda"]["items"][0]
    assert (first["title"], first["discussed_s"]) == ("Waitlist email, renamed", 25)
    assert saved_agenda(store, meeting).items[0].title == "Waitlist email, renamed"
    assert len(llm.calls) == 1


def test_an_edit_just_before_the_ticks_save_is_kept_without_asking_again(
    app, worker, client_as, store, model
):
    meeting, _ = standup(store, client_as)
    ingest(worker, meeting, said(meeting, 1, "Waitlist email", 0, 25))
    interfering = Interfering(store, lambda inner: rename_first(inner, meeting))
    app.dependency_overrides[get_store] = lambda: interfering
    model.says(current="a1")

    body = tracked(worker, meeting, now=30)

    assert interfering.conflicts == 1
    assert len(model.prompts) == 1
    saved = saved_agenda(store, meeting)
    assert (saved.items[0].title, saved.items[0].discussed_s) == ("Waitlist email, renamed", 25)
    assert saved.tracked_until == 30 - SETTLE_S
    assert body["agenda"] == saved.model_dump(mode="json")


def test_a_tick_that_keeps_losing_the_race_saves_nothing_and_is_retried(
    app, worker, client_as, store, model
):
    meeting, _ = standup(store, client_as)
    ingest(worker, meeting, said(meeting, 1, "Waitlist email", 0, 25))
    interfering = Interfering(store, lambda inner: rename_first(inner, meeting), times=99)
    app.dependency_overrides[get_store] = lambda: interfering
    model.says(current="a1")

    response = track(worker, meeting, now=30)

    assert response.status_code == 409
    assert saved_agenda(store, meeting).tracked_until is None


def test_a_lobby_edit_is_retried_on_top_of_a_tick_that_saved_first(app, client_as, store):
    meeting, (waitlist, refunds, _) = standup(store, client_as)

    async def tick(inner):
        agenda = await inner.agenda(meeting)
        items = [agenda.items[0].model_copy(update={"discussed_s": 40.0}), *agenda.items[1:]]
        await inner.save_agenda(
            agenda.model_copy(
                update={"items": items, "current_item_id": waitlist, "tracked_until": 55.0}
            )
        )

    interfering = Interfering(store, tick)
    app.dependency_overrides[get_store] = lambda: interfering

    response = client_as(SARAH).put(
        f"/meetings/{meeting}/agenda",
        json={"items": [{"id": refunds, "title": "Refunds"}, {"id": waitlist, "title": "Email"}]},
    )

    assert response.status_code == 200, response.text
    assert interfering.conflicts == 1
    saved = saved_agenda(store, meeting)
    assert [(i.title, i.discussed_s) for i in saved.items] == [("Refunds", 0), ("Email", 40)]
    assert (saved.current_item_id, saved.tracked_until) == (waitlist, 55)


def test_a_lobby_edit_that_keeps_losing_the_race_is_refused(app, client_as, store):
    meeting, (waitlist, *_) = standup(store, client_as)
    interfering = Interfering(store, lambda inner: rename_first(inner, meeting), times=99)
    app.dependency_overrides[get_store] = lambda: interfering

    response = client_as(SARAH).put(
        f"/meetings/{meeting}/agenda", json={"items": [{"id": waitlist, "title": "Email"}]}
    )

    assert response.status_code == 409
    assert saved_agenda(store, meeting).items[0].title == "Waitlist email, renamed"


# overlapping ticks


class SlowLLM:
    """Holds every classification until released, so two ticks overlap."""

    def __init__(self, inner: MockLLM):
        self.inner = inner
        self.last_model = inner.last_model
        self.released = False
        self.waiting = 0

    async def generate(self, prompt: str, *, system: str | None = None) -> str:
        raise NotImplementedError

    async def generate_structured(self, prompt, schema, *, system=None):
        self.waiting += 1
        while not self.released:
            await anyio.sleep(0.01)
        return await self.inner.generate_structured(prompt, schema, system=system)


def app_for(store, settings, llm):
    app = create_app()
    app.dependency_overrides[get_store] = lambda: store
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_llm_factory] = lambda: lambda: llm
    app.dependency_overrides[current_user] = lambda: ALEX
    return app


def brain_client(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://brain",
        headers={"X-Internal-Token": WORKER_TOKEN},
    )


async def overlapping_ticks(apps, meeting_id: str, slow: SlowLLM) -> list[httpx.Response]:
    clients = [brain_client(app) for app in apps]
    responses: list[httpx.Response] = []

    async def tick(client):
        url = f"/internal/meetings/{meeting_id}/agenda/track"
        responses.append(await client.post(url, json={"now": 60}))

    async with anyio.create_task_group() as tg:
        for client in clients:
            tg.start_soon(tick, client)
        with anyio.fail_after(5):
            while slow.waiting < 1:
                await anyio.sleep(0.01)
        await anyio.sleep(0.05)  # let the other tick reach the lock or the model
        slow.released = True
    for client in clients:
        await client.aclose()
    return responses


def seed_waitlist_talk(store) -> str:
    meeting = live_meeting(store).id
    seed(
        store,
        Agenda(
            meeting_id=meeting,
            items=[item("w", "Waitlist email", 10), item("r", "Refund policy", 5)],
            generated_at=T(),
        ),
    )
    talk = [TranscriptSegment(**said(meeting, 1, "Waitlist email", 0, 30))]
    anyio.run(store.add_segments, meeting, talk)
    return meeting


def test_two_overlapping_ticks_count_the_stretch_once(store, settings):
    meeting = seed_waitlist_talk(store)
    classifier = Classifier().says(current="a1").says(current="a1")
    slow = SlowLLM(classifier.llm)
    app = app_for(store, settings, slow)

    responses = anyio.run(overlapping_ticks, [app, app], meeting, slow)

    assert [r.status_code for r in responses] == [200, 200]
    assert len(classifier.prompts) == 1
    assert saved_agenda(store, meeting).items[0].discussed_s == 30


def test_two_replicas_ticking_at_once_count_the_stretch_once(store, settings):
    meeting = seed_waitlist_talk(store)
    classifier = Classifier().says(current="a1").says(current="a1")
    slow = SlowLLM(classifier.llm)
    replicas = [app_for(store, settings, slow), app_for(store, settings, slow)]

    responses = anyio.run(overlapping_ticks, replicas, meeting, slow)

    assert [r.status_code for r in responses] == [200, 200]
    assert saved_agenda(store, meeting).items[0].discussed_s == 30
    assert sorted(len(r.json()["nudges"]) for r in responses) == [0, 0]
    assert {r.json()["agenda"]["items"][0]["discussed_s"] for r in responses} == {30}


ROUNDS = 40


@pytest.mark.anyio
async def test_ticks_racing_lobby_edits_on_postgres_lose_no_edit(pg_dsn, settings):
    store = await postgres_world(pg_dsn)
    meeting = await store.create_meeting(TEAM.id, "Standup", ALEX.id)
    await store.update_meeting(
        meeting.model_copy(update={"started_at": datetime.now(UTC) - timedelta(hours=1)})
    )
    first = await store.save_agenda(
        Agenda(meeting_id=meeting.id, items=[item("w", "Waitlist email", 10)], generated_at=T())
    )
    llm = MockLLM(structured={AgendaTrackDraft: AgendaTrackDraft(current="a1")})
    app = app_for(store, settings, llm)
    statuses: list[tuple[int, int]] = []

    async with brain_client(app) as client:
        for n in range(ROUNDS):
            segment = said(meeting.id, n, f"Point {n}", 30 * n, 30 * n + 25)
            await store.add_segments(meeting.id, [TranscriptSegment(**segment)])
            replies: dict[str, httpx.Response] = {}

            async def tick(n=n, replies=replies):
                url = f"/internal/meetings/{meeting.id}/agenda/track"
                replies["tick"] = await client.post(url, json={"now": 30 * n + 35})

            async def edit(n=n, replies=replies):
                body = {"items": [{"id": "w", "title": f"Waitlist email {n}", "minutes": 10}]}
                replies["edit"] = await client.put(f"/meetings/{meeting.id}/agenda", json=body)

            async with anyio.create_task_group() as tg:
                tg.start_soon(tick)
                tg.start_soon(edit)
            statuses.append((replies["tick"].status_code, replies["edit"].status_code))
            saved = await store.agenda(meeting.id)
            assert saved.items[0].title == f"Waitlist email {n}", f"round {n} lost the edit"

    assert statuses == [(200, 200)] * ROUNDS
    saved = await store.agenda(meeting.id)
    assert saved.items[0].discussed_s == 30 * ROUNDS - 5
    assert saved.revision == first.revision + 2 * ROUNDS
