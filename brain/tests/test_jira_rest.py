"""Creating Jira issues through a site's REST API with a connected account."""

from datetime import date

import pytest
from jira_cloud_support import EMAIL, SITE, TOKEN, FakeJiraCloud, text_of
from test_jira_push import BOB, approve, draft, review

from brain.jira_rest import JiraAccess, JiraCloud, JiraRejected, JiraRestPusher, JiraUnreachable
from contracts import TaskPushResult

pytestmark = pytest.mark.anyio


def access(**changes) -> JiraAccess:
    return JiraAccess(
        **({"site": SITE, "email": EMAIL, "api_token": TOKEN, "project_key": "DS"} | changes)
    )


ACCESS = access()
BOB_AT_WORK = BOB.model_copy(update={"email": "bob@acme.example"})
BOB_IN_JIRA = {
    "accountId": "acc-bob",
    "displayName": "Bob Okafor",
    "emailAddress": "bob@acme.example",
}


async def push(jira: FakeJiraCloud, meeting, request, access: JiraAccess = ACCESS, **hooks):
    return await JiraRestPusher(access, transport=jira.transport).push(meeting, request, **hooks)


async def test_approved_drafts_become_issues_with_keys_and_links():
    jira = FakeJiraCloud()

    results = await push(jira, review(draft(1), draft(2)), approve(draft(1).id, draft(2).id))

    assert results == [
        TaskPushResult(task_id=draft(1).id, key="DS-1", url=f"https://{SITE}/browse/DS-1"),
        TaskPushResult(task_id=draft(2).id, key="DS-2", url=f"https://{SITE}/browse/DS-2"),
    ]
    first = jira.created[0]
    assert first["project"] == {"key": "DS"}
    assert first["issuetype"] == {"name": "Task"}
    assert first["summary"] == "Task 1"
    assert "assignee" not in first and "duedate" not in first


async def test_the_description_says_where_the_task_came_from_and_who_approved_it():
    jira = FakeJiraCloud()
    task = draft(1, description="Refund everyone charged twice.", owner_id=BOB.id)

    await push(jira, review(task), approve(task.id, approved_by="Alice Moreau"))

    description = jira.created[0]["description"]
    assert description["type"] == "doc" and description["version"] == 1
    assert [block["type"] for block in description["content"]] == [
        "paragraph",
        "paragraph",
        "blockquote",
        "paragraph",
        "paragraph",
    ]
    assert text_of(description).splitlines() == [
        "Refund everyone charged twice.",
        'From the meeting "Friday standup" (2026-10-02) at 00:06:',
        "I'll refund the 14 affected users by Wednesday.",
        "Owner named in the meeting: Bob Okafor",
        "Approved for Jira by Alice Moreau.",
    ]


async def test_an_owner_found_in_jira_is_assigned_and_a_due_date_is_set():
    jira = FakeJiraCloud(users=[BOB_IN_JIRA])
    task = draft(1, owner_id=BOB.id, due=date(2026, 10, 7))

    (result,) = await push(jira, review(task, members=[BOB_AT_WORK]), approve(task.id))

    assert result.warning is None
    assert jira.created[0]["assignee"] == {"id": "acc-bob"}
    assert jira.created[0]["duedate"] == "2026-10-07"
    lookup = next(r for r in jira.requests if r.url.path.endswith("/assignable/search"))
    assert dict(lookup.url.params) == {"project": "DS", "query": "bob@acme.example"}


async def test_an_owner_jira_does_not_know_leaves_the_issue_unassigned_with_a_warning():
    jira = FakeJiraCloud()
    task = draft(1, owner_id=BOB.id)

    (result,) = await push(jira, review(task, members=[BOB_AT_WORK]), approve(task.id))

    assert result.key == "DS-1"
    assert result.warning == "created unassigned: no Jira account matches bob@acme.example"
    assert "assignee" not in jira.created[0]


async def test_two_matching_accounts_leave_the_issue_unassigned():
    twin = {**BOB_IN_JIRA, "accountId": "acc-bob-2"}
    jira = FakeJiraCloud(users=[BOB_IN_JIRA, twin])
    task = draft(1, owner_id=BOB.id)

    (result,) = await push(jira, review(task, members=[BOB_AT_WORK]), approve(task.id))

    assert result.warning == "created unassigned: 2 Jira accounts match bob@acme.example"


async def test_a_lookalike_account_is_not_assigned():
    """Jira matches the start of a name, so "Bob" finds "Bobby Tables" too."""
    bobby = {
        "accountId": "acc-bobby",
        "displayName": "Bobby Tables",
        "emailAddress": "bt@x.example",
    }
    jira = FakeJiraCloud(users=[bobby])
    task = draft(1, owner_id=BOB.id)
    bob = BOB.model_copy(update={"name": "Bob", "email": None})

    (result,) = await push(jira, review(task, members=[bob]), approve(task.id))

    assert result.warning == "created unassigned: no Jira account matches Bob"
    assert "assignee" not in jira.created[0]


async def test_an_owner_found_by_exact_name_is_assigned():
    jira = FakeJiraCloud(users=[BOB_IN_JIRA])
    task = draft(1, owner_id=BOB.id)  # no email: looked up as "Bob Okafor"

    (result,) = await push(jira, review(task), approve(task.id))

    assert result.warning is None
    assert jira.created[0]["assignee"] == {"id": "acc-bob"}


async def test_an_owner_whose_email_jira_hides_is_assigned_on_the_one_match():
    jira = FakeJiraCloud(users=[BOB_IN_JIRA])
    jira.hide_emails = True
    task = draft(1, owner_id=BOB.id)

    (result,) = await push(jira, review(task, members=[BOB_AT_WORK]), approve(task.id))

    assert result.warning is None
    assert jira.created[0]["assignee"] == {"id": "acc-bob"}


async def test_the_summary_is_one_line_of_at_most_255_characters():
    jira = FakeJiraCloud()
    task = draft(1, title="Refund the users\n  charged twice " + "x" * 300)

    (result,) = await push(jira, review(task), approve(task.id))

    assert result.key == "DS-1"
    summary = jira.created[0]["summary"]
    assert summary.startswith("Refund the users charged twice x") and len(summary) == 255


async def test_the_issue_type_chosen_when_connecting_is_used():
    jira = FakeJiraCloud()

    await push(jira, review(draft(1)), approve(draft(1).id), access(issue_type_id="10003"))

    assert jira.created[0]["issuetype"] == {"id": "10003"}


async def test_each_issue_is_reported_as_soon_as_it_exists():
    jira = FakeJiraCloud()
    seen: list[tuple[str, int]] = []

    async def created(result):
        seen.append((result.key, len(jira.created)))

    await push(
        jira, review(draft(1), draft(2)), approve(draft(1).id, draft(2).id), on_created=created
    )

    assert seen == [("DS-1", 1), ("DS-2", 2)]


async def test_an_unexpected_failure_keeps_what_was_already_created():
    jira = FakeJiraCloud()
    jira.crash_on_create = 2

    first, second = await push(jira, review(draft(1), draft(2)), approve(draft(1).id, draft(2).id))

    assert (first.key, first.error) == ("DS-1", None)
    assert second.key is None and "unexpected" in second.error


async def test_a_create_jira_never_answered_says_the_issue_may_exist():
    jira = FakeJiraCloud()
    jira.time_out_on_create = 1

    first, second = await push(jira, review(draft(1), draft(2)), approve(draft(1).id, draft(2).id))

    assert first.key is None and "may have been created" in first.error
    assert "DS" in first.error  # where to look before pushing again
    assert second.key == "DS-1"  # the next draft still goes


async def test_a_redirect_is_never_followed():
    jira = FakeJiraCloud()
    jira.redirect = "https://elsewhere.example/rest/api/3/issue"

    (result,) = await push(jira, review(draft(1)), approve(draft(1).id))

    assert result.key is None and result.error
    assert {request.url.host for request in jira.requests} == {SITE}


async def test_a_field_the_project_cannot_set_is_left_out_and_said():
    jira = FakeJiraCloud()
    jira.unsettable = {"duedate"}
    task = draft(1, due=date(2026, 10, 7))

    (result,) = await push(jira, review(task), approve(task.id))

    assert result.key == "DS-1"
    assert result.warning == "created without a due date: this Jira project does not take one"
    assert "duedate" not in jira.created[0]


async def test_a_draft_jira_rejects_reports_jiras_reason_and_the_others_still_go():
    jira = FakeJiraCloud()
    bad = draft(1, title="FAIL this one")

    first, second = await push(jira, review(bad, draft(2)), approve(bad.id, draft(2).id))

    assert (first.key, first.error) == (None, "Jira refused it: Summary is not valid.")
    assert (second.key, second.error) == ("DS-1", None)


async def test_excluded_unknown_and_already_pushed_drafts_are_not_created_again():
    jira = FakeJiraCloud()
    pushed = draft(1, key="DS-90")
    excluded = draft(2, include=False)

    results = await push(
        jira, review(pushed, excluded), approve(pushed.id, excluded.id, "mtg-standup-task-9")
    )

    assert [(r.key, r.error) for r in results] == [
        ("DS-90", None),
        (None, "Draft is excluded"),
        (None, "No such task draft"),
    ]
    # Pushed before the account was connected, to wherever that was: not linked to this site.
    assert results[0].url is None
    assert jira.created == [] and jira.requests == []


async def test_a_draft_pushed_here_before_keeps_its_saved_link():
    jira = FakeJiraCloud()
    pushed = draft(1, key="DS-90", url=f"https://{SITE}/browse/DS-90")

    (result,) = await push(jira, review(pushed), approve(pushed.id))

    assert (result.key, result.url) == ("DS-90", f"https://{SITE}/browse/DS-90")
    assert jira.requests == []


async def test_a_wrong_token_says_to_connect_again_without_showing_it():
    jira = FakeJiraCloud()
    wrong = access(api_token="revoked-token")

    (result,) = await push(jira, review(draft(1)), approve(draft(1).id), wrong)

    assert result.key is None
    assert "connect Jira again" in result.error
    assert "revoked-token" not in result.error


async def test_an_unreachable_site_is_reported_per_draft():
    jira = FakeJiraCloud()
    jira.down = True

    results = await push(jira, review(draft(1), draft(2)), approve(draft(1).id, draft(2).id))

    assert [r.key for r in results] == [None, None]
    assert all(r.error.startswith(f"Could not reach {SITE}") for r in results)


async def test_pushing_elsewhere_than_jira_is_not_supported():
    jira = FakeJiraCloud()

    (result,) = await push(jira, review(draft(1)), approve(draft(1).id, destination="github"))

    assert result.error == "Pushing to github is not supported yet"
    assert jira.requests == []


async def test_the_account_and_its_project_can_be_checked_before_anything_is_saved():
    jira = FakeJiraCloud()
    cloud = JiraCloud(ACCESS, transport=jira.transport)

    assert (await cloud.myself())["displayName"] == "Acme Admin"
    assert (await cloud.project("DS"))["key"] == "DS"
    assert await cloud.can_create("DS") is True
    jira.can_create = False
    assert await cloud.can_create("DS") is False
    with pytest.raises(JiraRejected) as missing:
        await cloud.project("NOPE")
    assert missing.value.status == 404
    with pytest.raises(JiraRejected) as refused:
        await JiraCloud(access(api_token="x"), transport=jira.transport).myself()
    assert refused.value.status == 401
    jira.down = True
    with pytest.raises(JiraUnreachable):
        await cloud.myself()


def test_only_a_jira_cloud_site_is_accepted():
    for site in ("acme.atlassian.net", "https://Acme.atlassian.net/", "acme.atlassian.net/jira"):
        assert JiraAccess(site=site, email=EMAIL, api_token=TOKEN, project_key="DS").site == SITE
    for site in (
        "localhost:8080",
        "acme.example.com",
        "evil.com/acme.atlassian.net",
        "atlassian.net",
        "",
    ):
        with pytest.raises(ValueError):
            JiraAccess(site=site, email=EMAIL, api_token=TOKEN, project_key="DS")


def test_a_project_key_is_only_ever_a_key():
    for key in ("../myself", "DS/..", "ds", "D", "DS-1", ""):
        with pytest.raises(ValueError):
            access(project_key=key)


def test_the_token_does_not_show_when_the_access_is_printed():
    assert TOKEN not in repr(ACCESS) and TOKEN not in str(ACCESS)
    assert TOKEN not in ACCESS.model_dump_json()
