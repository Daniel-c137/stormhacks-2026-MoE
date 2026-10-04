"""The `Store` contract: what every store implementation must do, run against each one.

Every test here talks to the store only through the `Store` protocol and seeds its own data
through it, so the same tests run unchanged against any implementation.

Adding an implementation:

1. Write an async context manager that takes the test's fixture request, yields a fresh, empty
   store and cleans up after it. It can ask for fixtures it needs, as `postgres_store` does:

       @asynccontextmanager
       async def postgres_store(request) -> AsyncIterator[Store]:
           dsn = request.getfixturevalue("pg_dsn")
           ...
           yield PostgresStore(pool)

2. Add it to STORE_FACTORIES. A factory whose fixture skips (pgserver missing) skips its runs.

Every test then runs once per implementation (`-k memory`, `-k postgres` to pick one). Ids for
teams and people are UUID strings so a store with uuid columns accepts them.
"""

from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from datetime import UTC, date, datetime, timedelta
from uuid import uuid4

import anyio
import pytest

from brain.store import Conflict, FactCheckState, InMemoryStore, NotFound, ReportAudio, Store
from contracts import (
    AGENT_PARTICIPANT_ID,
    Agenda,
    AgendaItem,
    ChatMessage,
    Decision,
    DecisionRelation,
    FactCheck,
    GitHubSettings,
    JiraSettings,
    Person,
    Report,
    ReportProgress,
    Source,
    TaskDraft,
    Team,
    TeamSettings,
    TranscriptSegment,
)

pytestmark = pytest.mark.anyio


@asynccontextmanager
async def in_memory_store(request: pytest.FixtureRequest) -> AsyncIterator[Store]:
    yield InMemoryStore()


@asynccontextmanager
async def postgres_store(request: pytest.FixtureRequest) -> AsyncIterator[Store]:
    """A real Postgres (pgserver) with the migrations applied, through a connection pool."""
    from brain.db import migrate, open_pool
    from brain.pg_store import PostgresStore

    dsn = request.getfixturevalue("pg_dsn")
    await migrate(dsn)
    pool = await open_pool(dsn, max_size=4)
    try:
        yield PostgresStore(pool)
    finally:
        await pool.close()


StoreFactory = Callable[[pytest.FixtureRequest], AbstractAsyncContextManager[Store]]

STORE_FACTORIES = [
    pytest.param(in_memory_store, id="memory"),
    pytest.param(postgres_store, id="postgres"),
]


@pytest.fixture(params=STORE_FACTORIES)
async def store(request, anyio_backend) -> AsyncIterator[Store]:
    factory: StoreFactory = request.param
    async with factory(request) as store:
        yield store


T0 = datetime(2026, 10, 1, 16, 0, tzinfo=UTC)


def at(minutes: float) -> datetime:
    return T0 + timedelta(minutes=minutes)


def new_id() -> str:
    return str(uuid4())


def person(name: str, **fields) -> Person:
    words = name.split()
    initials = "".join(word[0] for word in words).upper()
    return Person(id=new_id(), name=name, short=words[0], initials=initials, **fields)


async def team_with(store: Store, *people: Person, name: str = "Checkout", **fields) -> Team:
    team = await store.create_team(Team(id=new_id(), name=name, member_ids=[], **fields))
    for p in people:
        await store.upsert_person(p, team.id)
    return await store.team(team.id)


async def two_teams(store: Store):
    """(team, alex, sarah, other_team, olga)"""
    alex = person("Alex Chen", email="alex@example.com", title="Backend engineer")
    sarah = person("Sarah Kim", email="sarah@example.com", title="Product manager")
    olga = person("Olga Petrova", email="olga@example.com", title="Backend engineer")
    team = await team_with(store, alex, sarah)
    other = await team_with(store, olga, name="Elsewhere")
    return team, alex, sarah, other, olga


async def ended_meeting(store: Store, team: Team, host: Person, title: str, start: datetime):
    meeting = await store.create_meeting(
        team.id, title, host.id, scheduled_start=start, duration_min=30
    )
    await store.start_meeting(meeting.id, start)
    end = start + timedelta(minutes=30)
    return await store.transition_status(meeting.id, {"live"}, "processing", at=end)


def segment(
    meeting_id: str, n: int, speaker: Person, *, t: float | None = None
) -> TranscriptSegment:
    t_start = float(n * 5) if t is None else t
    return TranscriptSegment(
        seg_id=f"{meeting_id}-{n}",
        meeting_id=meeting_id,
        speaker_id=speaker.id,
        speaker_name=speaker.name,
        text=f"{speaker.short} says thing {n}",
        is_final=True,
        t_start=t_start,
        t_end=t_start + 2.5,
    )


def chat(meeting_id: str, sender: Person, text: str, ts: datetime, **fields) -> ChatMessage:
    return ChatMessage(
        id=new_id(),
        meeting_id=meeting_id,
        sender_id=sender.id,
        sender_name=sender.name,
        is_agent=False,
        text=text,
        ts=ts,
        **fields,
    )


def report_for(meeting_id: str, *, owner: Person, decisions: list[str], tasks: list[str]) -> Report:
    return Report(
        meeting_id=meeting_id,
        summary="We agreed on the rollout.",
        topics=["rollout"],
        decisions=[
            Decision(
                id=f"{meeting_id}-decision-{i}",
                meeting_id=meeting_id,
                text=text,
                made_by=owner.name,
                t=float(i * 60),
                quote=text,
            )
            for i, text in enumerate(decisions, start=1)
        ],
        tasks=[
            TaskDraft(
                id=f"{meeting_id}-task-{i}",
                meeting_id=meeting_id,
                title=title,
                owner_id=owner.id if i == 1 else None,
                due=date(2026, 10, 8) if i == 1 else None,
                t=float(i * 60),
                quote=title,
            )
            for i, title in enumerate(tasks, start=1)
        ],
        open_questions=["Who tells support?"],
    )


# teams and people


async def test_a_created_team_can_be_read_back(store):
    team = await store.create_team(
        Team(id=new_id(), name="Checkout", member_ids=[], github_repo="acme/checkout")
    )

    assert await store.team(team.id) == team
    with pytest.raises(NotFound):
        await store.team(new_id())


async def test_upserting_a_person_adds_them_to_the_team_once(store):
    alex = person("Alex Chen", email="alex@example.com")
    team = await team_with(store, alex)

    renamed = alex.model_copy(update={"name": "Alex Chen-Ng"})
    await store.upsert_person(renamed, team.id)

    assert (await store.team(team.id)).member_ids == [alex.id]
    assert await store.members(team.id) == [renamed]
    assert await store.person(alex.id) == renamed


async def test_a_person_cannot_join_a_missing_team(store):
    with pytest.raises(NotFound):
        await store.upsert_person(person("Alex Chen"), new_id())


async def test_a_user_finds_their_own_team(store):
    team, alex, _, other, olga = await two_teams(store)

    assert (await store.team_for_user(alex.id)).id == team.id
    assert (await store.team_for_user(olga.id)).id == other.id
    with pytest.raises(NotFound):
        await store.team_for_user(new_id())


async def test_members_are_only_the_teams_own(store):
    team, alex, sarah, other, olga = await two_teams(store)

    assert {p.id for p in await store.members(team.id)} == {alex.id, sarah.id}
    assert [p.id for p in await store.members(other.id)] == [olga.id]
    with pytest.raises(NotFound):
        await store.members(new_id())


async def test_a_person_is_read_by_id(store):
    _, alex, *_ = await two_teams(store)

    assert await store.person(alex.id) == alex
    with pytest.raises(NotFound):
        await store.person(new_id())


async def test_a_profile_update_is_saved(store):
    _, alex, *_ = await two_teams(store)

    updated = alex.model_copy(update={"name": "Alex C.", "photo_url": "https://img.test/a.png"})
    assert await store.update_person(updated) == updated
    assert await store.person(alex.id) == updated

    with pytest.raises(NotFound):
        await store.update_person(person("Nobody Here"))


PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 64


async def test_a_saved_photo_is_served_from_the_persons_photo_path(store):
    _, alex, *_ = await two_teams(store)

    saved = await store.save_photo(alex.id, "image/png", PNG)

    assert saved.photo_url is not None
    assert saved.photo_url.split("?")[0] == f"/people/{alex.id}/photo"
    assert saved.name == alex.name
    assert await store.person(alex.id) == saved
    assert await store.photo(alex.id) == ("image/png", PNG)


async def test_a_new_photo_replaces_the_old_one_under_a_new_url(store):
    _, alex, *_ = await two_teams(store)
    first = await store.save_photo(alex.id, "image/png", PNG)

    second = await store.save_photo(alex.id, "image/jpeg", JPEG)

    assert second.photo_url != first.photo_url  # so a cached old photo is not shown
    assert await store.photo(alex.id) == ("image/jpeg", JPEG)


async def test_a_deleted_photo_is_gone(store):
    _, alex, sarah, *_ = await two_teams(store)
    await store.save_photo(alex.id, "image/png", PNG)

    cleared = await store.delete_photo(alex.id)

    assert cleared.photo_url is None
    assert (await store.person(alex.id)).photo_url is None
    with pytest.raises(NotFound):
        await store.photo(alex.id)
    assert (await store.delete_photo(sarah.id)).photo_url is None  # nothing to delete is fine


async def test_photos_need_an_existing_person(store):
    _, alex, *_ = await two_teams(store)

    with pytest.raises(NotFound):
        await store.photo(alex.id)  # no photo yet
    for missing in (store.photo(new_id()), store.delete_photo(new_id())):
        with pytest.raises(NotFound):
            await missing
    with pytest.raises(NotFound):
        await store.save_photo(new_id(), "image/png", PNG)


async def test_member_search_ignores_case_and_covers_name_email_and_title(store):
    team, alex, sarah, *_ = await two_teams(store)

    async def found(query: str) -> set[str]:
        return {p.id for p in await store.search_members(team.id, query)}

    assert await found("alex") == {alex.id}
    assert await found("KIM") == {sarah.id}
    assert await found("sarah@example") == {sarah.id}
    assert await found("product") == {sarah.id}
    assert await found("nobody") == set()


async def test_member_search_never_returns_another_teams_people(store):
    team, alex, _, other, olga = await two_teams(store)

    assert [p.id for p in await store.search_members(team.id, "backend")] == [alex.id]
    assert [p.id for p in await store.search_members(other.id, "backend")] == [olga.id]


async def test_member_search_is_capped_and_a_blank_query_lists_everyone(store):
    team = await team_with(store, *(person(f"Dev Number{i}") for i in range(5)))

    assert len(await store.search_members(team.id, "dev", limit=3)) == 3
    assert len(await store.search_members(team.id, "")) == 5


async def test_settings_default_to_the_teams_repo_and_project_until_saved(store):
    team = await team_with(store, name="Checkout", github_repo="acme/checkout", jira_project="DS")

    settings = await store.settings(team.id)

    assert settings == TeamSettings(
        team_id=team.id,
        github=GitHubSettings(repo="acme/checkout"),
        jira=JiraSettings(project="DS"),
    )
    assert settings.timezone == "UTC"
    with pytest.raises(NotFound):
        await store.settings(new_id())


async def test_saved_settings_are_read_back_per_team(store):
    team, _, _, other, _ = await two_teams(store)
    saved = TeamSettings(
        team_id=team.id,
        github=GitHubSettings(repo="acme/checkout", ref="main"),
        jira=JiraSettings(site="acme.atlassian.net", project="DS"),
        voice="voice-1",
        sensitivity="quiet",
        timezone="America/Vancouver",
    )

    assert await store.save_settings(saved) == saved

    assert await store.settings(team.id) == saved
    assert (await store.settings(other.id)).voice is None
    assert (await store.settings(other.id)).timezone == "UTC"
    with pytest.raises(NotFound):
        await store.save_settings(saved.model_copy(update={"team_id": new_id()}))


async def test_deleting_a_team_removes_everything_under_it_and_keeps_its_people(store):
    team, alex, sarah, other, olga = await two_teams(store)
    await store.save_settings(
        TeamSettings(team_id=team.id, github=GitHubSettings(), jira=JiraSettings(), voice="v")
    )
    meeting = await ended_meeting(store, team, alex, "Planning", at(0))
    await store.add_participant(meeting.id, sarah.id)
    await store.add_segments(meeting.id, [segment(meeting.id, 1, alex)])
    await store.add_public_chat(chat(meeting.id, alex, "hi", at(1)))
    await store.save_agenda(Agenda(meeting_id=meeting.id, items=[], generated_at=at(0)))
    report = report_for(meeting.id, owner=alex, decisions=["Use Redis"], tasks=["Ship it"])
    await store.complete_report(report)
    await store.save_report_progress(
        ReportProgress(meeting_id=meeting.id, steps=["Saving"], current=1, done=True)
    )
    await store.save_report_audio(meeting.id, "k", "audio/mpeg", b"mp3")
    kept = await ended_meeting(store, other, olga, "Elsewhere", at(0))
    await store.complete_report(report_for(kept.id, owner=olga, decisions=["A"], tasks=["B"]))

    await store.delete_team(team.id)

    with pytest.raises(NotFound):
        await store.team(team.id)
    with pytest.raises(NotFound):
        await store.settings(team.id)
    with pytest.raises(NotFound):
        await store.team_for_user(alex.id)
    with pytest.raises(NotFound):
        await store.meeting(meeting.id)
    with pytest.raises(NotFound):
        await store.report(meeting.id)
    assert await store.meetings(team.id) == []
    assert await store.transcript(meeting.id) == []
    assert await store.public_chat(meeting.id) == []
    assert await store.agenda(meeting.id) is None
    assert await store.report_progress(meeting.id) is None
    with pytest.raises(NotFound):
        await store.report_audio(meeting.id)
    with pytest.raises(NotFound):
        await store.task(team.id, report.tasks[0].id)
    assert await store.person(alex.id) == alex  # a person is not the team's
    assert await store.team(other.id) == other
    assert [m.id for m in await store.meetings(other.id)] == [kept.id]
    assert len(await store.decisions(other.id)) == len(await store.tasks(other.id)) == 1


async def test_a_deleted_team_can_be_created_again_empty(store):
    team, alex, *_ = await two_teams(store)
    await ended_meeting(store, team, alex, "Planning", at(0))

    await store.delete_team(team.id)
    again = await store.create_team(team.model_copy(update={"member_ids": []}))

    assert again.member_ids == []
    assert await store.meetings(team.id) == []
    await store.upsert_person(alex, team.id)
    assert (await store.team_for_user(alex.id)).id == team.id


async def test_deleting_a_missing_team_is_not_found(store):
    with pytest.raises(NotFound):
        await store.delete_team(new_id())


# meetings


async def test_an_unscheduled_meeting_starts_live_now(store):
    team, alex, *_ = await two_teams(store)
    before = datetime.now(UTC)

    meeting = await store.create_meeting(team.id, "Standup", alex.id)

    assert meeting.status == "live"
    assert meeting.team_id == team.id
    assert meeting.title == "Standup"
    assert meeting.host_id == alex.id
    assert meeting.code
    assert meeting.participant_ids == []
    assert meeting.invitee_ids == []
    assert meeting.scheduled_start is None
    assert before <= meeting.started_at <= datetime.now(UTC)
    assert meeting.ended_at is None
    assert meeting.transcript_deleted_at is None
    assert await store.meeting(meeting.id) == meeting


async def test_a_scheduled_meeting_waits_with_its_invitees(store):
    team, alex, sarah, *_ = await two_teams(store)

    meeting = await store.create_meeting(
        team.id,
        "Planning",
        alex.id,
        scheduled_start=at(60),
        duration_min=45,
        invitee_ids=[sarah.id],
    )

    assert meeting.status == "scheduled"
    assert meeting.scheduled_start == at(60)
    assert meeting.duration_min == 45
    assert meeting.invitee_ids == [sarah.id]
    assert meeting.started_at is None
    assert await store.meeting(meeting.id) == meeting


async def test_starting_a_scheduled_meeting_makes_it_live_once(store):
    team, alex, *_ = await two_teams(store)
    meeting = await store.create_meeting(team.id, "Planning", alex.id, scheduled_start=at(60))

    started = await store.start_meeting(meeting.id, at(62))
    again = await store.start_meeting(meeting.id, at(70))

    assert (started.status, started.started_at) == ("live", at(62))
    assert again == started
    assert await store.meeting(meeting.id) == started
    with pytest.raises(NotFound):
        await store.start_meeting(new_id(), at(0))


async def test_ending_a_live_meeting_records_when_and_happens_once(store):
    team, alex, *_ = await two_teams(store)
    meeting = await store.create_meeting(team.id, "Standup", alex.id)

    ended = await store.transition_status(meeting.id, {"live"}, "processing", at=at(30))

    assert (ended.status, ended.ended_at) == ("processing", at(30))
    assert ended.started_at == meeting.started_at
    assert await store.meeting(meeting.id) == ended
    with pytest.raises(Conflict):
        await store.transition_status(meeting.id, {"live"}, "processing", at=at(31))
    assert await store.meeting(meeting.id) == ended


async def test_ending_without_a_time_records_now(store):
    team, alex, *_ = await two_teams(store)
    meeting = await store.create_meeting(team.id, "Standup", alex.id)
    before = datetime.now(UTC)

    ended = await store.transition_status(meeting.id, {"live"}, "processing")

    assert before <= ended.ended_at <= datetime.now(UTC)


async def test_overlapping_ends_let_exactly_one_through(store):
    team, alex, *_ = await two_teams(store)
    meeting = await store.create_meeting(team.id, "Standup", alex.id)
    outcomes: list[str] = []

    async def end() -> None:
        try:
            await store.transition_status(meeting.id, {"live"}, "processing", at=at(30))
            outcomes.append("ended")
        except Conflict:
            outcomes.append("conflict")

    async with anyio.create_task_group() as tg:
        for _ in range(5):
            tg.start_soon(end)

    assert sorted(outcomes) == ["conflict"] * 4 + ["ended"]


async def test_a_transition_from_an_unexpected_status_is_refused(store):
    team, alex, *_ = await two_teams(store)
    meeting = await store.create_meeting(team.id, "Planning", alex.id, scheduled_start=at(60))

    with pytest.raises(Conflict):
        await store.transition_status(meeting.id, {"live"}, "processing")
    assert await store.meeting(meeting.id) == meeting

    reviewed = await store.transition_status(meeting.id, {"scheduled", "live"}, "live", at=at(61))
    assert (reviewed.status, reviewed.started_at, reviewed.ended_at) == ("live", at(61), None)
    with pytest.raises(NotFound):
        await store.transition_status(new_id(), {"live"}, "processing")


async def test_meetings_are_found_by_id_and_by_code(store):
    team, alex, *_ = await two_teams(store)
    first = await store.create_meeting(team.id, "Standup", alex.id)
    second = await store.create_meeting(team.id, "Retro", alex.id)

    assert first.code != second.code
    assert await store.meeting_by_code(second.code) == second
    with pytest.raises(NotFound):
        await store.meeting(new_id())
    with pytest.raises(NotFound):
        await store.meeting_by_code("no-such-code")


async def test_a_teams_meetings_are_its_own_newest_first(store):
    team, alex, _, other, olga = await two_teams(store)
    old = await ended_meeting(store, team, alex, "Old", at(0))
    upcoming = await store.create_meeting(team.id, "Upcoming", alex.id, scheduled_start=at(600))
    recent = await ended_meeting(store, team, alex, "Recent", at(120))
    await store.create_meeting(other.id, "Theirs", olga.id)

    assert [m.id for m in await store.meetings(team.id)] == [upcoming.id, recent.id, old.id]


async def test_status_changes_are_saved(store):
    team, alex, *_ = await two_teams(store)
    meeting = await store.create_meeting(team.id, "Standup", alex.id)

    processing = await store.set_status(meeting.id, "processing")

    assert processing.status == "processing"
    assert await store.meeting(meeting.id) == processing
    with pytest.raises(NotFound):
        await store.set_status(new_id(), "processing")


async def test_participants_are_added_once(store):
    team, alex, sarah, *_ = await two_teams(store)
    meeting = await store.create_meeting(team.id, "Standup", alex.id)

    await store.add_participant(meeting.id, alex.id)
    await store.add_participant(meeting.id, sarah.id)
    joined = await store.add_participant(meeting.id, alex.id)

    assert joined.participant_ids == [alex.id, sarah.id]
    assert (await store.meeting(meeting.id)).participant_ids == [alex.id, sarah.id]
    with pytest.raises(NotFound):
        await store.add_participant(new_id(), alex.id)


async def test_overlapping_joins_keep_every_participant(store):
    team = await team_with(store, *(person(f"Dev Number{i}") for i in range(6)))
    host, *others = await store.members(team.id)
    meeting = await store.create_meeting(team.id, "Standup", host.id)

    async with anyio.create_task_group() as tg:
        for p in others:
            tg.start_soon(store.add_participant, meeting.id, p.id)

    assert sorted((await store.meeting(meeting.id)).participant_ids) == sorted(p.id for p in others)


async def test_invitees_are_replaced_without_duplicates(store):
    team, alex, sarah, *_ = await two_teams(store)
    meeting = await store.create_meeting(
        team.id, "Planning", alex.id, scheduled_start=at(60), invitee_ids=[sarah.id]
    )

    invited = await store.set_invitees(meeting.id, [alex.id, sarah.id, alex.id])

    assert invited.invitee_ids == [alex.id, sarah.id]
    assert (await store.meeting(meeting.id)).invitee_ids == [alex.id, sarah.id]
    assert (await store.set_invitees(meeting.id, [])).invitee_ids == []
    with pytest.raises(NotFound):
        await store.set_invitees(new_id(), [alex.id])


async def test_meeting_updates_are_saved(store):
    team, alex, *_ = await two_teams(store)
    meeting = await store.create_meeting(team.id, "Standup", alex.id)

    updated = meeting.model_copy(
        update={"jira_keys": ["DS-117"], "ended_at": at(30), "title": "Daily standup"}
    )
    assert await store.update_meeting(updated) == updated
    assert await store.meeting(meeting.id) == updated

    with pytest.raises(NotFound):
        await store.update_meeting(updated.model_copy(update={"id": new_id()}))


# transcript


async def test_segments_are_saved_once_and_read_in_time_order(store):
    team, alex, sarah, *_ = await two_teams(store)
    meeting = await store.create_meeting(team.id, "Standup", alex.id)
    other = await store.create_meeting(team.id, "Retro", alex.id)
    late, early = segment(meeting.id, 2, sarah, t=30), segment(meeting.id, 1, alex, t=10)

    await store.add_segments(meeting.id, [late])
    await store.add_segments(meeting.id, [early, late])  # the worker resending after a timeout
    await store.add_segments(other.id, [segment(other.id, 1, alex)])

    assert await store.transcript(meeting.id) == [early, late]
    assert await store.transcript(new_id()) == []


async def test_a_seg_id_reused_with_different_content_is_a_conflict_and_saves_nothing(store):
    """A colliding id must be loud: silently keeping the first text loses the second."""
    team, alex, sarah, *_ = await two_teams(store)
    meeting = await store.create_meeting(team.id, "Standup", alex.id)
    first = segment(meeting.id, 1, alex, t=10)
    await store.add_segments(meeting.id, [first])
    fresh = segment(meeting.id, 2, sarah, t=20)

    with pytest.raises(Conflict):
        await store.add_segments(
            meeting.id, [fresh, first.model_copy(update={"text": "something else"})]
        )

    assert await store.transcript(meeting.id) == [first]


async def test_segments_at_the_same_time_are_ordered_by_seg_id(store):
    team, alex, sarah, *_ = await two_teams(store)
    meeting = await store.create_meeting(team.id, "Standup", alex.id)
    b = segment(meeting.id, 1, alex, t=10).model_copy(update={"seg_id": "b"})
    a = segment(meeting.id, 2, sarah, t=10).model_copy(update={"seg_id": "a"})

    await store.add_segments(meeting.id, [b, a])

    assert await store.transcript(meeting.id) == [a, b]


async def test_a_seg_id_is_only_a_duplicate_within_its_own_meeting(store):
    team, alex, *_ = await two_teams(store)
    first = await store.create_meeting(team.id, "Standup", alex.id)
    second = await store.create_meeting(team.id, "Retro", alex.id)
    mine = segment(first.id, 1, alex).model_copy(update={"seg_id": "seg-1"})
    theirs = segment(second.id, 1, alex).model_copy(update={"seg_id": "seg-1"})

    await store.add_segments(first.id, [mine])
    await store.add_segments(second.id, [theirs])

    assert await store.transcript(first.id) == [mine]
    assert await store.transcript(second.id) == [theirs]


async def test_segments_for_a_missing_meeting_are_refused(store):
    _, alex, *_ = await two_teams(store)
    missing = new_id()

    with pytest.raises(NotFound):
        await store.add_segments(missing, [segment(missing, 1, alex)])


async def test_deleting_a_transcript_drops_its_segments_and_records_when(store):
    team, alex, *_ = await two_teams(store)
    meeting = await store.create_meeting(team.id, "Standup", alex.id)
    kept = await store.create_meeting(team.id, "Retro", alex.id)
    await store.add_segments(meeting.id, [segment(meeting.id, 1, alex)])
    await store.add_segments(kept.id, [segment(kept.id, 1, alex)])

    deleted = await store.delete_transcript(meeting.id, at(90))

    assert deleted.transcript_deleted_at == at(90)
    assert (await store.meeting(meeting.id)).transcript_deleted_at == at(90)
    assert await store.transcript(meeting.id) == []
    assert len(await store.transcript(kept.id)) == 1
    with pytest.raises(NotFound):
        await store.delete_transcript(new_id(), at(90))


async def test_transcripts_due_for_deletion_are_ended_before_the_cutoff_and_still_there(store):
    team, alex, _, other, olga = await two_teams(store)
    due = await ended_meeting(store, team, alex, "Due", at(0))  # ended at(30)
    due_elsewhere = await ended_meeting(store, other, olga, "Theirs", at(10))
    too_recent = await ended_meeting(store, team, alex, "Recent", at(100))
    already_deleted = await ended_meeting(store, team, alex, "Gone", at(0))
    no_transcript = await ended_meeting(store, team, alex, "Silent", at(0))
    still_live = await store.create_meeting(team.id, "Live", alex.id)
    for m in (due, due_elsewhere, too_recent, already_deleted, still_live):
        await store.add_segments(m.id, [segment(m.id, 1, alex)])
    await store.delete_transcript(already_deleted.id, at(50))

    found = await store.meetings_with_transcript_before(at(60))

    assert {m.id for m in found} == {due.id, due_elsewhere.id}
    assert no_transcript.id not in {m.id for m in found}


# public chat


async def test_public_chat_is_saved_once_in_time_order(store):
    team, alex, sarah, *_ = await two_teams(store)
    meeting = await store.create_meeting(team.id, "Standup", alex.id)
    other = await store.create_meeting(team.id, "Retro", alex.id)
    later = chat(meeting.id, sarah, "Link: DS-117", at(5))
    sooner = chat(meeting.id, alex, "Morning", at(1))

    await store.add_public_chat(later)
    await store.add_public_chat(sooner)
    await store.add_public_chat(later)
    await store.add_public_chat(chat(other.id, alex, "Elsewhere", at(2)))

    assert await store.public_chat(meeting.id) == [sooner, later]


async def test_private_chat_is_never_stored(store):
    team, alex, *_ = await two_teams(store)
    meeting = await store.create_meeting(team.id, "Standup", alex.id)
    private = chat(meeting.id, alex, "Is DS-117 done?", at(1), visibility="private")
    addressed = chat(
        meeting.id, alex, "Only for the agent", at(2), recipient_id=AGENT_PARTICIPANT_ID
    )

    with pytest.raises(ValueError):
        await store.add_public_chat(private)
    with pytest.raises(ValueError):
        await store.add_public_chat(addressed)

    assert await store.public_chat(meeting.id) == []


async def test_public_chat_for_a_missing_meeting_is_refused(store):
    _, alex, *_ = await two_teams(store)

    with pytest.raises(NotFound):
        await store.add_public_chat(chat(new_id(), alex, "Hello?", at(0)))


# agenda


async def test_an_agenda_is_none_until_saved_then_replaced_on_save(store):
    team, alex, sarah, *_ = await two_teams(store)
    meeting = await store.create_meeting(team.id, "Planning", alex.id, scheduled_start=at(60))
    assert await store.agenda(meeting.id) is None

    first = Agenda(
        meeting_id=meeting.id,
        items=[AgendaItem(id="a1", title="Refund status", minutes=10, added_by=alex.id)],
        generated_at=at(0),
    )
    saved = await store.save_agenda(first)
    assert saved == first.model_copy(update={"revision": 1})
    assert await store.agenda(meeting.id) == saved

    edited = first.model_copy(
        update={
            "items": [
                AgendaItem(id="a2", title="Rollout plan", minutes=5, added_by=sarah.id),
                first.items[0],
            ],
            "updated_at": at(10),
        }
    )
    await store.save_agenda(edited)

    assert await store.agenda(meeting.id) == edited.model_copy(update={"revision": 2})


async def test_an_agendas_timekeeping_state_is_saved_with_it(store):
    team, alex, *_ = await two_teams(store)
    meeting = await store.create_meeting(team.id, "Standup", alex.id)
    tracked = Agenda(
        meeting_id=meeting.id,
        items=[
            AgendaItem(id="a1", title="Waitlist email", minutes=10, discussed_s=312.5),
            AgendaItem(id="a2", title="Refund policy", minutes=5, nudged_t=1260.0),
        ],
        generated_at=at(0),
        current_item_id="a1",
        tracked_until=1265.25,
    )

    await store.save_agenda(tracked)

    saved = await store.agenda(meeting.id)
    assert saved == tracked.model_copy(update={"revision": 1})
    assert (saved.current_item_id, saved.tracked_until) == ("a1", 1265.25)
    assert [(i.discussed_s, i.nudged_t) for i in saved.items] == [(312.5, None), (0, 1260.0)]


async def test_a_conditional_save_goes_through_only_at_the_revision_it_read(store):
    team, alex, *_ = await two_teams(store)
    meeting = await store.create_meeting(team.id, "Standup", alex.id)
    planned = Agenda(
        meeting_id=meeting.id,
        items=[AgendaItem(id="a1", title="Waitlist email", minutes=10)],
        generated_at=at(0),
    )
    with pytest.raises(NotFound):
        await store.save_agenda_if(planned.model_copy(update={"meeting_id": new_id()}))

    # Revision 0 is "none saved yet": the first conditional save creates it.
    created = await store.save_agenda_if(planned)
    assert created == planned.model_copy(update={"revision": 1})
    with pytest.raises(Conflict):
        await store.save_agenda_if(planned)

    tracked = created.model_copy(update={"current_item_id": "a1", "tracked_until": 55.0})
    second = await store.save_agenda_if(tracked)
    assert second == tracked.model_copy(update={"revision": 2})
    renamed = created.model_copy(
        update={"items": [created.items[0].model_copy(update={"title": "Email"})]}
    )
    with pytest.raises(Conflict):  # read at revision 1; the tracked save landed since
        await store.save_agenda_if(renamed)
    assert await store.agenda(meeting.id) == second

    await store.save_agenda(second)  # an unconditional save bumps the revision too
    with pytest.raises(Conflict):
        await store.save_agenda_if(second)
    assert (await store.agenda(meeting.id)).revision == 3


async def test_overlapping_conditional_saves_let_exactly_one_through(store):
    team, alex, *_ = await two_teams(store)
    meeting = await store.create_meeting(team.id, "Standup", alex.id)
    planned = Agenda(
        meeting_id=meeting.id,
        items=[AgendaItem(id="a1", title="Waitlist email", minutes=10)],
        generated_at=at(0),
    )
    outcomes: list[str] = []

    async def save(agenda: Agenda) -> None:
        try:
            await store.save_agenda_if(agenda)
            outcomes.append("saved")
        except Conflict:
            outcomes.append("conflict")

    async with anyio.create_task_group() as tg:  # five first saves
        for n in range(5):
            tg.start_soon(save, planned.model_copy(update={"tracked_until": float(n)}))
    assert sorted(outcomes) == ["conflict"] * 4 + ["saved"]

    read = await store.agenda(meeting.id)
    outcomes.clear()
    async with anyio.create_task_group() as tg:  # five saves of what was read at revision 1
        for n in range(5):
            tg.start_soon(save, read.model_copy(update={"tracked_until": 55.0 + n}))
    assert sorted(outcomes) == ["conflict"] * 4 + ["saved"]
    assert (await store.agenda(meeting.id)).revision == 2


async def test_overlapping_unconditional_saves_each_bump_the_revision(store):
    team, alex, *_ = await two_teams(store)
    meeting = await store.create_meeting(team.id, "Standup", alex.id)
    planned = Agenda(meeting_id=meeting.id, items=[], generated_at=at(0))

    async with anyio.create_task_group() as tg:
        for _ in range(5):
            tg.start_soon(store.save_agenda, planned)

    assert (await store.agenda(meeting.id)).revision == 5


async def test_an_agenda_for_a_missing_meeting_is_refused(store):
    with pytest.raises(NotFound):
        await store.save_agenda(Agenda(meeting_id=new_id(), items=[], generated_at=at(0)))


# fact-checks


def fact_check(claim: str, **fields) -> FactCheck:
    defaults = {"verdict": "contradicted", "confidence": 0.875, "severity": "high"}
    return FactCheck(id=new_id(), claim=claim, speaker_name="Sarah Kim", **(defaults | fields))


async def test_public_fact_checks_are_kept_once_in_the_order_they_were_added(store):
    team, alex, *_ = await two_teams(store)
    meeting = await store.create_meeting(team.id, "Standup", alex.id)
    other = await store.create_meeting(team.id, "Planning", alex.id)
    assert await store.fact_checks(meeting.id) == []

    released = fact_check(
        "PR 41 is released.",
        sources=[
            Source(kind="github_pr", label="dropsubs/app#41", url="https://x.test/pull/41"),
            Source(kind="github_release", label="dropsubs/app@v0.9.3"),
        ],
        snippet_ids=["snip-1"],
        raised_hand=True,
        t=12.5,
        created_at=at(1),
    )
    closed = fact_check("DS-104 is closed.", verdict="supported", severity="low", t=3.0)
    await store.add_fact_check(meeting.id, released)
    await store.add_fact_check(meeting.id, closed)
    await store.add_fact_check(meeting.id, released.model_copy(update={"claim": "changed"}))

    assert await store.fact_checks(meeting.id) == [released, closed]
    assert await store.fact_checks(other.id) == []


async def test_a_private_fact_check_is_never_stored(store):
    team, alex, sarah, *_ = await two_teams(store)
    meeting = await store.create_meeting(team.id, "Standup", alex.id)

    with pytest.raises(ValueError):
        await store.add_fact_check(
            meeting.id, fact_check("x", visibility="private", recipient_id=sarah.id)
        )
    with pytest.raises(ValueError):
        await store.add_fact_check(meeting.id, fact_check("x", recipient_id=sarah.id))

    assert await store.fact_checks(meeting.id) == []


async def test_a_fact_check_for_a_missing_meeting_is_refused(store):
    with pytest.raises(NotFound):
        await store.add_fact_check(new_id(), fact_check("x"))
    with pytest.raises(NotFound):
        await store.save_fact_check_state_if(
            FactCheckState(meeting_id=new_id(), checked_until=5.0), checked_until=None
        )


async def test_fact_check_state_saves_only_from_the_expected_point(store):
    team, alex, *_ = await two_teams(store)
    meeting = await store.create_meeting(team.id, "Standup", alex.id)
    assert await store.fact_check_state(meeting.id) is None

    first = FactCheckState(meeting_id=meeting.id, checked_until=55.0)
    assert await store.save_fact_check_state_if(first, checked_until=None) == first
    assert await store.fact_check_state(meeting.id) == first
    with pytest.raises(Conflict):
        await store.save_fact_check_state_if(
            first.model_copy(update={"checked_until": 85.0}), checked_until=None
        )
    with pytest.raises(Conflict):
        await store.save_fact_check_state_if(
            first.model_copy(update={"checked_until": 85.0}), checked_until=50.0
        )

    second = FactCheckState(
        meeting_id=meeting.id, checked_until=115.0, checked_at=120.0, hand_raised_at=120.0
    )
    assert await store.save_fact_check_state_if(second, checked_until=55.0) == second
    assert await store.fact_check_state(meeting.id) == second


async def test_overlapping_fact_check_state_saves_let_exactly_one_through(store):
    team, alex, *_ = await two_teams(store)
    meeting = await store.create_meeting(team.id, "Standup", alex.id)
    outcomes: list[str] = []

    async def save(n: int) -> None:
        state = FactCheckState(meeting_id=meeting.id, checked_until=55.0 + n, checked_at=60.0)
        try:
            await store.save_fact_check_state_if(state, checked_until=None)
            outcomes.append("saved")
        except Conflict:
            outcomes.append("conflict")

    async with anyio.create_task_group() as tg:
        for n in range(5):
            tg.start_soon(save, n)

    assert sorted(outcomes) == ["conflict"] * 4 + ["saved"]


# reports, tasks and decisions


async def test_a_report_is_read_back_after_it_is_saved(store):
    team, alex, *_ = await two_teams(store)
    meeting = await ended_meeting(store, team, alex, "Standup", at(0))
    with pytest.raises(NotFound):
        await store.report(meeting.id)

    report = report_for(meeting.id, owner=alex, decisions=["Ship Friday"], tasks=["Refund users"])
    await store.save_report(report)

    assert await store.report(meeting.id) == report
    with pytest.raises(NotFound):
        await store.save_report(report.model_copy(update={"meeting_id": new_id()}))


async def test_saving_a_report_makes_its_tasks_and_decisions_queryable(store):
    team, alex, _, other, olga = await two_teams(store)
    meeting = await ended_meeting(store, team, alex, "Standup", at(0))
    theirs = await ended_meeting(store, other, olga, "Theirs", at(0))
    report = report_for(meeting.id, owner=alex, decisions=["Ship Friday"], tasks=["Refund", "Docs"])
    await store.save_report(report)
    await store.save_report(report_for(theirs.id, owner=olga, decisions=["X"], tasks=["Y"]))

    assert [t.id for t in await store.tasks(team.id)] == [t.id for t in report.tasks]
    assert [d.id for d in await store.decisions(team.id)] == [d.id for d in report.decisions]
    assert await store.task(team.id, report.tasks[0].id) == report.tasks[0]
    with pytest.raises(NotFound):
        await store.task(other.id, report.tasks[0].id)
    with pytest.raises(NotFound):
        await store.task(team.id, new_id())


async def test_tasks_can_be_filtered_by_owner(store):
    team, alex, sarah, *_ = await two_teams(store)
    meeting = await ended_meeting(store, team, alex, "Standup", at(0))
    report = report_for(meeting.id, owner=alex, decisions=[], tasks=["Refund", "Docs"])
    await store.save_report(report)

    assert [t.id for t in await store.tasks(team.id, owner_id=alex.id)] == [report.tasks[0].id]
    assert await store.tasks(team.id, owner_id=sarah.id) == []


async def test_an_edited_task_is_saved_and_shows_in_the_report(store):
    team, alex, sarah, *_ = await two_teams(store)
    meeting = await ended_meeting(store, team, alex, "Standup", at(0))
    report = report_for(meeting.id, owner=alex, decisions=[], tasks=["Refund", "Docs"])
    await store.save_report(report)

    edited = report.tasks[1].model_copy(
        update={"title": "Write the docs", "owner_id": sarah.id, "include": False}
    )
    assert await store.update_task(edited) == edited

    assert await store.task(team.id, edited.id) == edited
    assert (await store.report(meeting.id)).tasks == [report.tasks[0], edited]
    assert [t.id for t in await store.tasks(team.id, owner_id=sarah.id)] == [edited.id]
    with pytest.raises(NotFound):
        await store.update_task(edited.model_copy(update={"id": new_id()}))


async def test_decisions_are_newest_first_and_searchable(store):
    team, alex, *_ = await two_teams(store)
    older = await ended_meeting(store, team, alex, "Older", at(0))
    newer = await ended_meeting(store, team, alex, "Newer", at(120))
    await store.save_report(
        report_for(older.id, owner=alex, decisions=["Use Redis for sessions"], tasks=[])
    )
    await store.save_report(
        report_for(newer.id, owner=alex, decisions=["Ship Friday", "Drop redis"], tasks=[])
    )

    texts = [d.text for d in await store.decisions(team.id)]
    found = [d.text for d in await store.decisions(team.id, query="REDIS")]

    assert texts == ["Drop redis", "Ship Friday", "Use Redis for sessions"]
    assert found == ["Drop redis", "Use Redis for sessions"]


async def test_decisions_are_only_the_teams_own(store):
    team, _, _, other, olga = await two_teams(store)
    theirs = await ended_meeting(store, other, olga, "Theirs", at(0))
    await store.save_report(report_for(theirs.id, owner=olga, decisions=["Secret"], tasks=[]))

    assert await store.decisions(team.id) == []
    assert await store.decisions(team.id, query="secret") == []
    assert await store.tasks(team.id) == []


async def test_an_updated_decision_is_saved(store):
    team, alex, *_ = await two_teams(store)
    meeting = await ended_meeting(store, team, alex, "Standup", at(0))
    report = report_for(meeting.id, owner=alex, decisions=["Ship Friday"], tasks=[])
    await store.save_report(report)

    superseded = report.decisions[0].model_copy(update={"status": "superseded"})
    assert await store.update_decision(superseded) == superseded

    assert await store.decisions(team.id) == [superseded]
    assert (await store.report(meeting.id)).decisions == [superseded]
    with pytest.raises(NotFound):
        await store.update_decision(superseded.model_copy(update={"id": new_id()}))


async def test_saving_a_report_again_replaces_its_tasks_and_decisions(store):
    team, alex, *_ = await two_teams(store)
    meeting = await ended_meeting(store, team, alex, "Standup", at(0))
    await store.save_report(
        report_for(meeting.id, owner=alex, decisions=["A", "B"], tasks=["X", "Y"])
    )

    redo = report_for(meeting.id, owner=alex, decisions=["C"], tasks=["Z"])
    redo = redo.model_copy(
        update={
            "decisions": [redo.decisions[0].model_copy(update={"id": f"{meeting.id}-d-new"})],
            "tasks": [redo.tasks[0].model_copy(update={"id": f"{meeting.id}-t-new"})],
        }
    )
    await store.save_report(redo)

    assert await store.report(meeting.id) == redo
    assert [d.text for d in await store.decisions(team.id)] == ["C"]
    assert [t.title for t in await store.tasks(team.id)] == ["Z"]
    with pytest.raises(NotFound):
        await store.task(team.id, f"{meeting.id}-task-2")


def superseded_by(past: Decision, new: Decision) -> Decision:
    relation = DecisionRelation(type="superseded_by", decision_id=new.id)
    return past.model_copy(update={"status": "superseded", "relation": relation})


async def linked_meetings(store: Store):
    """(team, alex, an earlier meeting's report with one decision, a processing meeting)"""
    team, alex, *_ = await two_teams(store)
    earlier = await ended_meeting(store, team, alex, "Planning", at(0))
    past = report_for(earlier.id, owner=alex, decisions=["Use Redis for jobs"], tasks=[])
    await store.save_report(past)
    meeting = await ended_meeting(store, team, alex, "Review", at(120))
    return team, alex, past, meeting


async def test_completing_a_report_saves_it_applies_its_links_and_moves_it_to_review(store):
    team, alex, past, meeting = await linked_meetings(store)
    report = report_for(meeting.id, owner=alex, decisions=["Keep Postgres for jobs"], tasks=["X"])
    (old,), (new,) = past.decisions, report.decisions
    contradicts = DecisionRelation(type="contradicts", decision_id=old.id)
    report.decisions[0] = new = new.model_copy(update={"relation": contradicts})

    completed = await store.complete_report(report, [superseded_by(old, new)])

    assert completed.status == "needs_review"
    assert await store.meeting(meeting.id) == completed
    assert await store.report(meeting.id) == report
    assert {d.id: d for d in await store.decisions(team.id)} == {
        old.id: superseded_by(old, new),
        new.id: new,
    }


async def test_completing_a_report_is_all_or_nothing(store):
    team, alex, past, meeting = await linked_meetings(store)
    report = report_for(meeting.id, owner=alex, decisions=["Keep Postgres"], tasks=["X"])
    (old,), (new,) = past.decisions, report.decisions
    missing = superseded_by(old.model_copy(update={"id": new_id()}), new)

    with pytest.raises(NotFound):
        await store.complete_report(report, [superseded_by(old, new), missing])

    assert (await store.meeting(meeting.id)).status == "processing"
    with pytest.raises(NotFound):
        await store.report(meeting.id)
    assert await store.decisions(team.id) == [old]
    assert await store.tasks(team.id) == []


async def test_only_a_meeting_being_written_up_can_be_completed(store):
    team, alex, *_ = await two_teams(store)
    live = await store.create_meeting(team.id, "Live", alex.id)
    report = report_for(live.id, owner=alex, decisions=["A"], tasks=[])

    with pytest.raises(Conflict):
        await store.complete_report(report)
    assert (await store.meeting(live.id)).status == "live"
    with pytest.raises(NotFound):
        await store.report(live.id)

    meeting = await ended_meeting(store, team, alex, "Review", at(0))
    await store.complete_report(report_for(meeting.id, owner=alex, decisions=[], tasks=[]))
    with pytest.raises(Conflict):
        await store.complete_report(report_for(meeting.id, owner=alex, decisions=["B"], tasks=[]))
    with pytest.raises(NotFound):
        await store.complete_report(report.model_copy(update={"meeting_id": new_id()}))


async def test_completing_again_undoes_the_links_of_the_meetings_earlier_decisions(store):
    team, alex, past, meeting = await linked_meetings(store)
    other = await ended_meeting(store, team, alex, "Other", at(60))
    theirs = report_for(other.id, owner=alex, decisions=["Use Kafka"], tasks=[])
    await store.save_report(theirs)
    (old,) = past.decisions
    # An earlier, interrupted save left this meeting's decision retiring the past one.
    earlier = report_for(meeting.id, owner=alex, decisions=["Keep Postgres"], tasks=[])
    await store.save_report(earlier)
    await store.update_decision(superseded_by(old, earlier.decisions[0]))
    # A decision retired by another meeting's decision is not this meeting's to undo.
    await store.update_decision(superseded_by(theirs.decisions[0], old))

    await store.complete_report(report_for(meeting.id, owner=alex, decisions=[], tasks=[]))

    saved = {d.id: d for d in await store.decisions(team.id)}
    assert saved[old.id] == old
    assert saved[theirs.decisions[0].id] == superseded_by(theirs.decisions[0], old)


async def test_report_progress_is_none_until_saved_and_keeps_its_error(store):
    team, alex, *_ = await two_teams(store)
    meeting = await ended_meeting(store, team, alex, "Standup", at(0))
    assert await store.report_progress(meeting.id) is None

    steps = ["Transcript", "Summary", "Tasks"]
    running = ReportProgress(meeting_id=meeting.id, steps=steps, current=1, done=False)
    await store.save_report_progress(running)
    assert await store.report_progress(meeting.id) == running

    failed = running.model_copy(
        update={"current": 2, "error": "Gemini is unavailable", "updated_at": at(31)}
    )
    await store.save_report_progress(failed)
    assert await store.report_progress(meeting.id) == failed

    with pytest.raises(NotFound):
        await store.save_report_progress(running.model_copy(update={"meeting_id": new_id()}))


# the report read aloud

MP3 = b"ID3\x04\x00\x00\x00\x00\x00\x00" + b"\xff\xfb" * 32


async def test_report_audio_is_not_found_until_saved_then_read_back(store):
    team, alex, *_ = await two_teams(store)
    meeting = await ended_meeting(store, team, alex, "Standup", at(0))
    with pytest.raises(NotFound):
        await store.report_audio(meeting.id)

    saved = await store.save_report_audio(meeting.id, "key-1", "audio/mpeg", MP3)

    assert saved == ReportAudio(key="key-1", content_type="audio/mpeg", data=MP3)
    assert await store.report_audio(meeting.id) == saved


async def test_saving_report_audio_again_replaces_it(store):
    team, alex, *_ = await two_teams(store)
    meeting = await ended_meeting(store, team, alex, "Standup", at(0))
    await store.save_report_audio(meeting.id, "key-1", "audio/mpeg", MP3)

    await store.save_report_audio(meeting.id, "key-2", "audio/mpeg", b"other audio")

    assert await store.report_audio(meeting.id) == ReportAudio(
        key="key-2", content_type="audio/mpeg", data=b"other audio"
    )


async def test_report_audio_belongs_to_its_meeting(store):
    team, alex, *_ = await two_teams(store)
    first = await ended_meeting(store, team, alex, "Standup", at(0))
    second = await ended_meeting(store, team, alex, "Retro", at(60))
    await store.save_report_audio(first.id, "key-1", "audio/mpeg", MP3)

    with pytest.raises(NotFound):
        await store.report_audio(second.id)
    with pytest.raises(NotFound):
        await store.report_audio(new_id())
    with pytest.raises(NotFound):
        await store.save_report_audio(new_id(), "key-1", "audio/mpeg", MP3)
