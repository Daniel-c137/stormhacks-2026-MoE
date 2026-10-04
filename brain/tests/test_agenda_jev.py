"""Keeping time against the agenda with Jev (JEV_MODEL): one request per stretch asks which agenda
item each new line is about and how likely each pending item is over. Jev is fast and cheap
enough to ask as soon as a caption settles, so a finished item is ticked within seconds; without
it the tracker asks Gemini at its own pace, as before. Either way, a stretch of only small talk
never finishes an item."""

import anyio
import pytest
from api_support import SARAH
from test_agenda_tracking import (
    Classifier,
    by_id,
    covered_by,
    ingest,
    said,
    saved_agenda,
    seed,
    standup,
    track,
    tracked,
)

from brain.agent.agenda_jev import COVERED_P, JEV_MAX_LINES
from brain.agent.timekeeping import JEV_SETTLE_S, JEV_TRANSLATION_LAG_S, SETTLE_S
from brain.api.deps import get_jev, get_llm_factory
from brain.llm import LLMError
from contracts import AGENT_PARTICIPANT_ID


class FakeJev:
    """Answers each request with the next scripted reply: each new line about `about[n]` (the
    last label repeated for any more lines) and each pending item over with the given chance."""

    model = "typesafe/jev-test"

    def __init__(self):
        self.calls: list[tuple[dict, dict]] = []
        self.replies: list[tuple[tuple[str, ...], dict[str, float]]] = []
        self.error: Exception | None = None

    def says(self, *about: str, closed: dict[str, float] | None = None) -> "FakeJev":
        self.replies.append((about or ("none",), closed or {}))
        return self

    async def decide(self, state: dict, questions: dict) -> dict:
        self.calls.append((state, questions))
        if self.error:
            raise self.error
        about, closed = self.replies.pop(0) if self.replies else (("none",), {})
        lines = sorted((k for k in questions if k.startswith("about_")), key=line_number)
        answers: dict[str, dict] = {}
        for n, key in enumerate(lines):
            label = about[min(n, len(about) - 1)]
            answers[key] = {"choice": label, "probabilities": {label: 0.9}, "confidence": 0.9}
        for key in questions:
            if key.startswith("closed_"):
                answers[key] = {"noul": closed.get(key.removeprefix("closed_"), 0.05)}
        return answers


def line_number(key: str) -> int:
    return int(key.split("_", 1)[1])


@pytest.fixture
def jev(app) -> FakeJev:
    fake = FakeJev()
    app.dependency_overrides[get_jev] = lambda: fake
    return fake


@pytest.fixture
def gemini(app) -> Classifier:
    """Gemini as the tracker would call it; with Jev on it is never asked."""
    classifier = Classifier()
    app.dependency_overrides[get_llm_factory] = lambda: lambda: classifier.llm
    return classifier


# asked as soon as a caption settles


def test_a_short_closing_line_is_ticked_as_soon_as_it_settles(
    worker, client_as, store, jev, gemini
):
    meeting, (waitlist, *_) = standup(store, client_as)
    ingest(worker, meeting, said(meeting, 1, "Waitlist is done, next", 10, 12))
    jev.says("a1", closed={"a1": 0.9})

    body = tracked(worker, meeting, now=12 + JEV_SETTLE_S)

    assert JEV_SETTLE_S < SETTLE_S
    assert covered_by(by_id(body["agenda"])[waitlist]) == (AGENT_PARTICIPANT_ID, 12)
    assert body["agenda"]["tracked_until"] == 12
    assert len(jev.calls) == 1
    assert gemini.prompts == []


def test_a_line_that_has_not_settled_waits(worker, client_as, store, jev):
    meeting, _ = standup(store, client_as)
    ingest(worker, meeting, said(meeting, 1, "Waitlist is done, next", 10, 12))

    body = tracked(worker, meeting, now=12 + JEV_SETTLE_S - 0.5)

    assert jev.calls == []
    assert body["agenda"]["tracked_until"] is None


def test_another_speakers_caption_arriving_late_is_still_asked_about(worker, client_as, store, jev):
    """Sarah's caption ended before Alex's but reached the brain 2.5 s after hers ended: it is
    not skipped, because Alex's had not settled yet when the check after it ran."""
    meeting, _ = standup(store, client_as)
    ingest(worker, meeting, said(meeting, 1, "Waitlist email goes Thursday", 4, 10))
    jev.says("a1")

    tracked(worker, meeting, now=12.5)  # the worker's check after Alex's caption, too early
    ingest(worker, meeting, said(meeting, 2, "Agreed, that's settled", 7, 9.5, SARAH))
    tracked(worker, meeting, now=10 + JEV_SETTLE_S)

    assert JEV_SETTLE_S >= 3
    [(state, _)] = jev.calls
    assert list(state["new_lines"].values()) == [
        "[00:04] Alex Chen: Waitlist email goes Thursday",
        "[00:07] Sarah Kim: Agreed, that's settled",
    ]


def test_with_translation_on_a_caption_settles_after_its_translation(worker, client_as, store, jev):
    meeting, _ = standup(store, client_as)
    live = anyio.run(store.meeting, meeting)
    anyio.run(store.update_meeting, live.model_copy(update={"translate": True}))
    ingest(worker, meeting, said(meeting, 1, "Waitlist is done, next", 10, 12))
    jev.says("a1", closed={"a1": 0.9})

    early = tracked(worker, meeting, now=12 + JEV_SETTLE_S)
    settled = tracked(worker, meeting, now=12 + JEV_SETTLE_S + JEV_TRANSLATION_LAG_S)

    assert early["agenda"]["tracked_until"] is None
    assert settled["agenda"]["tracked_until"] == 12
    assert len(jev.calls) == 1


def test_nothing_new_asks_nothing(worker, client_as, store, jev):
    meeting, _ = standup(store, client_as)
    ingest(worker, meeting, said(meeting, 1, "Waitlist email goes out Thursday", 0, 10))
    jev.says("a1")
    tracked(worker, meeting, now=20)

    tracked(worker, meeting, now=30)

    assert len(jev.calls) == 1


# what Jev is asked


def test_jev_is_asked_about_each_new_line_and_each_pending_item(worker, client_as, store, jev):
    meeting, (_, _, launch) = standup(store, client_as)
    agenda = saved_agenda(store, meeting)
    done = [
        i.model_copy(update={"status": "covered"}) if i.id == launch else i for i in agenda.items
    ]
    seed(store, agenda.model_copy(update={"items": done}))
    ingest(worker, meeting, said(meeting, 1, "The waitlist email draft is ready", 0, 8))
    jev.says("a1")
    tracked(worker, meeting, now=20)
    ingest(
        worker,
        meeting,
        said(meeting, 2, "I'd cut the second paragraph", 30, 36, SARAH),
        said(meeting, 3, "Fine, then refunds", 37, 40),
    )
    jev.says("a1", "a2")

    tracked(worker, meeting, now=50)

    state, questions = jev.calls[-1]
    assert sorted(k for k in questions if k.startswith("about_")) == ["about_1", "about_2"]
    for key in ("about_1", "about_2"):
        assert questions[key]["type"] == "choice"
        assert set(questions[key]["criteria"]) == {"a1", "a2", "a3", "other", "none"}
    assert "Waitlist email" in questions["about_1"]["criteria"]["a1"]
    assert sorted(k for k in questions if k.startswith("closed_")) == ["closed_a1", "closed_a2"]
    assert questions["closed_a1"]["type"] == "noul"
    assert "Waitlist email" in questions["closed_a1"]["instructions"]
    assert state["new_lines"] == {
        "1": "[00:30] Sarah Kim: I'd cut the second paragraph",
        "2": "[00:37] Alex Chen: Fine, then refunds",
    }
    assert state["earlier_lines"] == ["[00:00] Alex Chen: The waitlist email draft is ready"]
    assert state["being_discussed_before"] == "a1"
    assert "discussed 8 s" in state["agenda"]["a1"] and "covered" in state["agenda"]["a3"]


def test_jevs_answers_label_each_line_and_tick_the_items_it_finds_over(
    worker, client_as, store, jev
):
    meeting, (waitlist, refunds, _) = standup(store, client_as)
    ingest(
        worker,
        meeting,
        said(meeting, 1, "Waitlist email went out, that's done", 0, 8),
        said(meeting, 2, "Refunds: annual plans are the open question", 10, 20, SARAH),
    )
    jev.says("a1", "a2", closed={"a1": 0.8, "a2": 0.3})

    body = tracked(worker, meeting, now=30)

    items = by_id(body["agenda"])
    assert covered_by(items[waitlist]) == (AGENT_PARTICIPANT_ID, 8)
    assert items[refunds]["status"] == "pending"
    assert body["agenda"]["current_item_id"] == refunds
    assert (items[waitlist]["discussed_s"], items[refunds]["discussed_s"]) == (10, 10)


def test_an_item_jev_is_unsure_about_stays_pending(worker, client_as, store, jev):
    meeting, (waitlist, *_) = standup(store, client_as)
    ingest(worker, meeting, said(meeting, 1, "The waitlist email is nearly there", 0, 8))
    jev.says("a1", closed={"a1": COVERED_P - 0.01})

    body = tracked(worker, meeting, now=20)

    assert by_id(body["agenda"])[waitlist]["status"] == "pending"


def test_a_label_jev_was_not_offered_counts_for_no_item(worker, client_as, store, jev):
    meeting, (waitlist, *_) = standup(store, client_as)
    ingest(worker, meeting, said(meeting, 1, "The waitlist email is nearly there", 0, 8))
    jev.says("a9")

    body = tracked(worker, meeting, now=20)

    assert body["agenda"]["tracked_until"] == 8
    assert by_id(body["agenda"])[waitlist]["discussed_s"] == 0


def test_a_long_backlog_is_asked_about_in_parts(worker, client_as, store, jev):
    meeting, (waitlist, *_) = standup(store, client_as)
    lines = JEV_MAX_LINES + 5
    ingest(
        worker,
        meeting,
        *(said(meeting, n, f"Waitlist point {n}", 2 * n, 2 * n + 1) for n in range(lines)),
    )
    jev.says("a1").says("a1")

    body = tracked(worker, meeting, now=2 * lines + 10)

    first, second = jev.calls
    assert len([k for k in first[1] if k.startswith("about_")]) == JEV_MAX_LINES
    assert len([k for k in second[1] if k.startswith("about_")]) == 5
    assert second[0]["earlier_lines"][-1].endswith(f"Waitlist point {JEV_MAX_LINES - 1}")
    assert body["agenda"]["tracked_until"] == 2 * (lines - 1) + 1
    assert by_id(body["agenda"])[waitlist]["discussed_s"] == 2 * (lines - 1) + 1


# small talk never finishes an item


def test_small_talk_after_an_item_does_not_tick_it(worker, client_as, store, jev):
    meeting, (waitlist, *_) = standup(store, client_as)
    ingest(worker, meeting, said(meeting, 1, "The waitlist email draft is ready", 0, 30))
    jev.says("a1")
    tracked(worker, meeting, now=40)
    ingest(worker, meeting, said(meeting, 2, "Did anyone watch the game last night?", 41, 44))
    jev.says("none", closed={"a1": 0.75})

    body = tracked(worker, meeting, now=50)

    assert by_id(body["agenda"])[waitlist]["status"] == "pending"


def test_moving_on_to_work_off_the_agenda_ticks_the_item(worker, client_as, store, jev):
    """Off-agenda work is not small talk: leaving an item for it finishes the item."""
    meeting, (waitlist, *_) = standup(store, client_as)
    ingest(worker, meeting, said(meeting, 1, "The waitlist email draft is ready", 0, 30))
    jev.says("a1")
    tracked(worker, meeting, now=40)
    ingest(worker, meeting, said(meeting, 2, "Also, we need to pick a NoSQL database", 41, 44))
    jev.says("other", closed={"a1": 0.75})

    body = tracked(worker, meeting, now=50)

    items = by_id(body["agenda"])
    assert covered_by(items[waitlist]) == (AGENT_PARTICIPANT_ID, 30)
    assert body["agenda"]["current_item_id"] is None  # nothing on the agenda is being discussed


def test_small_talk_after_an_item_does_not_tick_it_with_gemini_either(
    worker, client_as, store, gemini
):
    meeting, (waitlist, *_) = standup(store, client_as)
    ingest(worker, meeting, said(meeting, 1, "The waitlist email draft is ready", 0, 30))
    gemini.says("a1")
    tracked(worker, meeting, now=40)
    ingest(worker, meeting, said(meeting, 2, "Did anyone watch the game last night?", 41, 50))
    gemini.says("none", covered=("a1",))

    body = tracked(worker, meeting, now=60)

    assert len(gemini.prompts) == 2
    assert by_id(body["agenda"])[waitlist]["status"] == "pending"


# when Jev is off or fails


def test_when_jev_fails_nothing_is_tracked_and_gemini_is_not_asked(
    worker, client_as, store, jev, gemini
):
    meeting, _ = standup(store, client_as)
    ingest(worker, meeting, said(meeting, 1, "Waitlist is done, next", 10, 12))
    jev.error = LLMError("Jev answered 503: overloaded")

    response = track(worker, meeting, now=20)

    assert response.status_code == 502
    assert "overloaded" in response.json()["detail"]
    assert response.json()["agenda"]["tracked_until"] is None
    assert gemini.prompts == []


def test_without_jev_gemini_is_asked_at_its_own_pace(worker, client_as, store, gemini, app):
    app.dependency_overrides[get_jev] = lambda: None
    meeting, _ = standup(store, client_as)
    ingest(worker, meeting, said(meeting, 1, "Waitlist is done, next", 10, 12))
    gemini.says("a1", covered=("a1",))

    early = tracked(worker, meeting, now=12 + SETTLE_S)

    assert gemini.prompts == []  # settled, but too little talk and not waited long: as before
    assert early["agenda"]["tracked_until"] is None
