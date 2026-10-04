from datetime import date

import pytest

from brain.config import Settings
from brain.llm import LLMError, MockLLM, make_llm
from brain.memory import Chunk, MemoryHit
from brain.report.decisions import (
    MAX_CANDIDATES,
    DecisionVerdict,
    DecisionVerdicts,
    apply_links,
    link_decisions,
    memory_candidates,
)
from brain.store import InMemoryStore
from contracts import Decision, DecisionRelation, Report, Team

pytestmark = pytest.mark.anyio

SEP_18, OCT_2 = date(2026, 9, 18), date(2026, 10, 2)


def decision(id: str, meeting_id: str, text: str, **fields) -> Decision:
    fields = {"made_by": "Alice Moreau", "t": 60, "quote": text, **fields}
    return Decision(id=id, meeting_id=meeting_id, text=text, **fields)


REDIS = decision(
    "d-redis",
    "m-sep",
    "Move the job queue from Postgres to Redis before Black Friday",
    quote="Let's move the job queue over to Redis before Black Friday.",
)
KEEP_PG = decision(
    "d-keep-pg",
    "m-oct",
    "Keep Postgres for the job queue and revisit above a few hundred jobs a minute",
    quote="We keep the queue on Postgres; we revisit if we pass a few hundred jobs a minute.",
)
SEARCH = decision(
    "d-search",
    "m-oct",
    "Use Postgres full-text search for the help center instead of Algolia",
)
DATES = {"m-sep": SEP_18, "m-oct": OCT_2}


def verdicts(*items: DecisionVerdict) -> MockLLM:
    return MockLLM(structured={DecisionVerdicts: DecisionVerdicts(verdicts=list(items))})


def verdict(new: Decision, past: Decision | str | None, kind="contradicts", confidence=0.9):
    return DecisionVerdict(
        decision_id=new.id,
        verdict=kind,
        past_decision_id=past.id if isinstance(past, Decision) else past,
        reason="It reverses the earlier plan.",
        confidence=confidence,
    )


async def test_the_job_queue_reversal_links_both_ways():
    llm = verdicts(verdict(KEEP_PG, REDIS))

    links = await link_decisions(llm, [KEEP_PG], [REDIS], dates=DATES)

    assert links.new == [
        KEEP_PG.model_copy(
            update={"relation": DecisionRelation(type="contradicts", decision_id="d-redis")}
        )
    ]
    assert links.past == [
        REDIS.model_copy(
            update={
                "status": "superseded",
                "relation": DecisionRelation(type="superseded_by", decision_id="d-keep-pg"),
            }
        )
    ]
    (call,) = llm.calls
    for shown in (REDIS.quote, KEEP_PG.quote, "2026-09-18", "2026-10-02", "d-redis"):
        assert shown in call.prompt


async def test_a_superseding_decision_retires_the_old_one():
    llm = verdicts(verdict(KEEP_PG, REDIS, kind="supersedes"))

    links = await link_decisions(llm, [KEEP_PG], [REDIS], dates=DATES)

    assert links.new == []  # the contract has no "supersedes" relation on the newer decision
    assert [(d.id, d.status, d.relation) for d in links.past] == [
        ("d-redis", "superseded", DecisionRelation(type="superseded_by", decision_id="d-keep-pg"))
    ]


async def test_an_unrelated_pair_does_not_link():
    llm = verdicts(verdict(SEARCH, REDIS, kind="unrelated", confidence=0.95))

    links = await link_decisions(llm, [SEARCH], [REDIS], dates=DATES)

    assert len(llm.calls) == 1  # both mention Postgres, so the model is asked
    assert (links.new, links.past) == ([], [])


async def test_a_low_confidence_verdict_does_not_link():
    llm = verdicts(verdict(KEEP_PG, REDIS, confidence=0.5))

    links = await link_decisions(llm, [KEEP_PG], [REDIS], dates=DATES)

    assert (links.new, links.past) == ([], [])


async def test_a_candidate_id_the_model_was_not_given_is_ignored():
    unshown = decision("d-unshown", "m-sep", "Hire a second designer")
    llm = verdicts(
        verdict(KEEP_PG, "d-made-up"),
        verdict(KEEP_PG, unshown),  # real, but no overlap, so never offered
        verdict(decision("d-ghost", "m-oct", "x"), REDIS),  # not one of the new decisions
    )

    links = await link_decisions(llm, [KEEP_PG], [REDIS, unshown], dates=DATES)

    assert "d-unshown" not in llm.calls[0].prompt
    assert (links.new, links.past) == ([], [])


async def test_decisions_from_the_same_meeting_or_itself_are_never_linked():
    same_meeting = decision("d-same", "m-oct", "Move the job queue to Redis next sprint")
    llm = verdicts(verdict(KEEP_PG, same_meeting), verdict(KEEP_PG, KEEP_PG))

    links = await link_decisions(llm, [KEEP_PG], [same_meeting, KEEP_PG], dates=DATES)

    assert llm.calls == []  # no candidate is left, so nothing is asked
    assert (links.new, links.past) == ([], [])


async def test_only_active_past_decisions_are_candidates():
    retired = REDIS.model_copy(update={"status": "superseded"})
    llm = verdicts(verdict(KEEP_PG, retired))

    links = await link_decisions(llm, [KEEP_PG], [retired], dates=DATES)

    assert llm.calls == []
    assert (links.new, links.past) == ([], [])


async def test_no_past_decisions_makes_no_llm_call():
    llm = verdicts()

    links = await link_decisions(llm, [KEEP_PG, SEARCH], [])

    assert llm.calls == []
    assert (links.new, links.past) == ([], [])


async def test_candidates_are_bounded_to_the_closest_past_decisions():
    close = [
        decision(f"d-close-{i}", "m-sep", f"Keep the job queue on Postgres option {i}")
        for i in range(MAX_CANDIDATES)
    ]
    far = decision("d-far", "m-sep", "Postgres backups run nightly")
    llm = verdicts()

    await link_decisions(llm, [KEEP_PG], [far, *close])

    (call,) = llm.calls
    assert all(d.id in call.prompt for d in close)
    assert "d-far" not in call.prompt


async def test_a_past_decision_is_superseded_once_by_the_most_confident_link():
    also = decision("d-also", "m-oct", "Keep the job queue in Postgres through Black Friday")
    llm = verdicts(verdict(also, REDIS, confidence=0.8), verdict(KEEP_PG, REDIS, confidence=0.95))

    links = await link_decisions(llm, [also, KEEP_PG], [REDIS], dates=DATES)

    assert [d.id for d in links.new] == ["d-keep-pg"]
    assert links.past[0].relation == DecisionRelation(type="superseded_by", decision_id="d-keep-pg")


async def test_apply_links_saves_both_sides():
    store = InMemoryStore()
    team = await store.create_team(Team(id="t", name="Checkout", member_ids=[]))
    sep = await store.create_meeting(team.id, "Queue planning", "p-alice")
    oct_ = await store.create_meeting(team.id, "Queue review", "p-alice")
    redis = REDIS.model_copy(update={"meeting_id": sep.id})
    keep = KEEP_PG.model_copy(update={"meeting_id": oct_.id})
    await store.save_report(Report(meeting_id=sep.id, summary="", decisions=[redis]))
    await store.save_report(Report(meeting_id=oct_.id, summary="", decisions=[keep]))

    links = await link_decisions(verdicts(verdict(keep, redis)), [keep], [redis])
    await apply_links(store, links)

    saved = {d.id: d for d in await store.decisions(team.id)}
    assert saved["d-redis"].status == "superseded"
    assert saved["d-redis"].relation == DecisionRelation(type="superseded_by", decision_id=keep.id)
    assert saved["d-keep-pg"].relation == DecisionRelation(type="contradicts", decision_id=redis.id)


# candidates from meeting memory


class FakeMemory:
    """Answers each search with the chunks scripted for that query, best first."""

    def __init__(self, results: dict[str, list[Chunk]]):
        self.results = results
        self.searches: list[tuple[str, str, int]] = []

    async def search(self, team_id: str, query: str, k: int = 8) -> list[MemoryHit]:
        self.searches.append((team_id, query, k))
        chunks = self.results.get(query, [])[:k]
        return [MemoryHit(chunk=c, score=1 - i / 10) for i, c in enumerate(chunks)]


def chunk(kind, ref_id: str | None = None, meeting_id: str = "m-sep") -> Chunk:
    return Chunk(
        id=f"{meeting_id}:{kind}:{ref_id}",
        team_id="t",
        meeting_id=meeting_id,
        kind=kind,
        text="...",
        ref_id=ref_id,
    )


BETA = decision("d-beta", "m-sep", "Announce the beta to everyone on Monday")
HIRE = decision("d-hire", "m-sep", "Hire a second designer")


async def test_memory_hits_become_candidates_in_memory_order_mapped_by_ref_id():
    memory = FakeMemory(
        {
            KEEP_PG.text: [
                chunk("transcript", None),
                chunk("decision", "d-hire"),
                chunk("task", "d-beta"),  # a task chunk is never a decision candidate
                chunk("decision", "d-unknown"),  # not one of the past decisions given
                chunk("decision", "d-beta"),
            ]
        }
    )

    pick = await memory_candidates(memory, "t", [KEEP_PG], [REDIS, BETA, HIRE])

    assert [d.id for d in pick(KEEP_PG, [REDIS, BETA, HIRE])][:2] == ["d-hire", "d-beta"]
    (search,) = memory.searches
    assert search[:2] == ("t", KEEP_PG.text)


async def test_lexical_candidates_fill_in_after_the_memory_hits():
    memory = FakeMemory({KEEP_PG.text: [chunk("decision", "d-hire")]})

    pick = await memory_candidates(memory, "t", [KEEP_PG], [REDIS, HIRE, BETA])

    # d-redis shares words with the new decision; d-beta shares none and memory missed it
    assert [d.id for d in pick(KEEP_PG, [REDIS, HIRE, BETA])] == ["d-hire", "d-redis"]


async def test_without_memory_hits_the_candidates_are_lexical():
    pick = await memory_candidates(FakeMemory({}), "t", [KEEP_PG], [REDIS, BETA])

    assert pick(KEEP_PG, [REDIS, BETA]) == [REDIS]


async def test_memory_candidates_keep_to_the_eligible_decisions_and_the_limit():
    many = [decision(f"d-{i}", "m-sep", f"Unrelated choice {i}") for i in range(MAX_CANDIDATES + 2)]
    memory = FakeMemory({KEEP_PG.text: [chunk("decision", d.id) for d in [HIRE, *many]]})

    pick = await memory_candidates(memory, "t", [KEEP_PG], [HIRE, *many])

    shown = pick(KEEP_PG, many)  # d-hire is not eligible for this call
    assert [d.id for d in shown] == [d.id for d in many[:MAX_CANDIDATES]]


class BrokenMemory:
    async def search(self, team_id: str, query: str, k: int = 8) -> list[MemoryHit]:
        raise LLMError("Gemini embeddings are unavailable on every model tried")


async def test_a_memory_search_failure_falls_back_to_lexical_candidates(caplog):
    pick = await memory_candidates(BrokenMemory(), "t", [KEEP_PG], [REDIS, BETA])

    assert pick(KEEP_PG, [REDIS, BETA]) == [REDIS]
    assert "lexical" in caplog.text


async def test_a_past_decision_found_only_in_memory_reaches_the_model():
    memory = FakeMemory({KEEP_PG.text: [chunk("decision", "d-beta")]})
    llm = verdicts()

    pick = await memory_candidates(memory, "t", [KEEP_PG], [BETA])
    await link_decisions(llm, [KEEP_PG], [BETA], candidates=pick)

    (call,) = llm.calls
    assert "d-beta" in call.prompt


settings = Settings()


@pytest.mark.live
@pytest.mark.skipif(
    not (settings.gemini_api_key and settings.gemini_model),
    reason="set GEMINI_API_KEY and GEMINI_MODEL to run against Gemini",
)
async def test_gemini_links_only_the_job_queue_reversal():
    llm = make_llm(settings)

    links = await link_decisions(llm, [KEEP_PG, SEARCH], [REDIS], dates=DATES)
    print(f"answered by {llm.last_model}")

    assert [(d.id, d.status, d.relation) for d in links.past] == [
        ("d-redis", "superseded", DecisionRelation(type="superseded_by", decision_id="d-keep-pg"))
    ]
    # "contradicts" is expected; a "supersedes" verdict leaves the newer decision unchanged
    assert [d.id for d in links.new] in (["d-keep-pg"], [])
