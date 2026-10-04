"""The mock GitHub server answers issue, pull request, commit and release reads from
mock-data/github/ in the shapes of GitHub's official MCP server, scopes searches like its
prepareSearchArgs, refuses what it does not support, and journals comments to the overlay."""

import json
from datetime import datetime

import pytest
from mcp import Client
from mcp.types import TextContent

from world import overlay
from world.config import MOCK_DATA_DIR, SNAPSHOTS_DIR, world_spec
from world.github_mcp import server
from world.overlay import JournalOverlay
from world.snapshot import mock_records

pytestmark = pytest.mark.anyio

FULL_NAME = world_spec().github_repo
OWNER, REPO = FULL_NAME.split("/")
WEB = f"https://github.com/{FULL_NAME}"
DATA = MOCK_DATA_DIR / "github"


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(autouse=True)
def overlay_dir(tmp_path, monkeypatch):
    """Each test writes to its own overlay, over the demo snapshot."""
    monkeypatch.setenv("WORLD_OVERLAY_DIR", str(tmp_path / "overlay"))
    monkeypatch.setenv("WORLD_SNAPSHOT", "demo")
    return tmp_path / "overlay"


def raw(kind: str) -> list[dict]:
    return json.loads((DATA / f"{kind}.json").read_text())


def head(snapshot: str) -> str:
    return json.loads((SNAPSHOTS_DIR / snapshot / "repo.json").read_text())["sha"]


async def call(tool: str, **arguments):
    async with Client(server) as client:
        return await client.call_tool(tool, arguments)


async def read(tool: str, **arguments):
    """A read of the world's repository: owner and repo filled in."""
    return payload(await call(tool, owner=OWNER, repo=REPO, **arguments))


def payload(result):
    assert not result.is_error, text(result)
    if result.structured_content is not None:
        return result.structured_content
    return json.loads(text(result))


def text(result) -> str:
    return "".join(c.text for c in result.content if isinstance(c, TextContent))


def numbers(items: list[dict]) -> list[int]:
    return [item["number"] for item in items]


def at(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


async def search(tool: str, query: str, **arguments) -> list[int]:
    return numbers(payload(await call(tool, query=query, **arguments))["items"])


# One repository


def test_the_mock_data_and_the_code_describe_the_worlds_repository():
    for kind in ("issues", "pull_requests", "releases", "commits", "comments"):
        for record in raw(kind):
            assert record["html_url"].startswith(f"{WEB}/"), (kind, record["html_url"])
    shas = [c["sha"] for c in raw("commits")]
    assert head("dev") in shas and head("demo") in shas
    assert shas[0] == head("demo")  # the demo's main is the newest commit


async def test_the_server_offers_the_official_read_tools():
    async with Client(server) as client:
        tools = {tool.name for tool in (await client.list_tools()).tools}

    assert {
        "list_issues",
        "search_issues",
        "issue_read",
        "list_pull_requests",
        "search_pull_requests",
        "pull_request_read",
        "list_commits",
        "list_releases",
        "get_latest_release",
        "search_code",
        "get_file_contents",
        "add_issue_comment",
    } <= tools


# search_issues and search_pull_requests


async def test_the_brains_issue_search_finds_the_repos_issues_in_the_rest_shape():
    data = payload(await call("search_issues", query="charged twice", owner=OWNER, repo=REPO))

    assert data["total_count"] == 1 and data["incomplete_results"] is False
    (item,) = data["items"]
    (expected,) = [i for i in raw("issues") if i["number"] == 43]
    assert item["title"] == expected["title"] == "Subscriptions show up twice; fee charged twice"
    assert item["state"] == "closed"
    assert item["html_url"] == f"{WEB}/issues/43"
    assert item["repository_url"] == f"https://api.github.com/repos/{FULL_NAME}"
    assert item["labels"] == expected["labels"]
    assert "pull_request" not in item


async def test_issue_search_needs_every_word_as_a_whole_word_ignoring_case():
    assert await search("search_issues", "GMAIL") == [10, 1]  # newest update first
    assert await search("search_issues", "gmail nonexistentword") == []
    assert await search("search_issues", "gma") == []


async def test_comments_count_as_text():
    # issue 37 says "fee" only in a comment
    (issue,) = [i for i in raw("issues") if i["number"] == 37]
    assert "fee" not in f"{issue['title']} {issue['body']}".casefold().split()

    assert 37 in await search("search_issues", "fee")


async def test_issue_search_leaves_out_pull_requests_and_pr_search_finds_only_them():
    assert await search("search_issues", "Gmail") == [10, 1]
    assert await search("search_pull_requests", "Gmail") == [2]


async def test_a_pull_request_search_item_carries_its_merge_time_in_pull_request():
    data = payload(
        await call("search_pull_requests", query="Approve check", owner=OWNER, repo=REPO)
    )

    assert numbers(data["items"]) == [50, 27]
    item = data["items"][0]
    assert item["html_url"] == f"{WEB}/pull/50"
    assert item["pull_request"]["merged_at"] == "2026-10-02T23:40:00Z"
    assert item["pull_request"]["html_url"] == f"{WEB}/pull/50"


@pytest.mark.parametrize(
    ("query", "owner", "repo", "found"),
    [
        ("Gmail", OWNER, REPO, [10, 1]),
        ("Gmail", OWNER, None, [10, 1]),  # owner alone does not scope, as in prepareSearchArgs
        ("Gmail", "otherorg", "private-repo", []),
        ("Gmail repo:otherorg/private-repo", OWNER, REPO, []),  # the query's repo: wins
        (f"Gmail repo:{FULL_NAME.upper()}", "otherorg", "private-repo", [10, 1]),
        (f"Gmail org:{OWNER}", None, None, [10, 1]),
        (f"Gmail user:{OWNER}", None, None, [10, 1]),
        ("Gmail org:otherorg", None, None, []),
        (f"Gmail repo:otherorg/private-repo repo:{FULL_NAME}", None, None, [10, 1]),
    ],
)
async def test_searches_scope_like_prepare_search_args(query, owner, repo, found):
    arguments = {k: v for k, v in {"owner": owner, "repo": repo}.items() if v}

    assert await search("search_issues", query, **arguments) == found


@pytest.mark.parametrize(
    ("tool", "query", "found"),
    [
        ("search_issues", "waitlist is:open", [52]),
        ("search_issues", "waitlist is:closed", [9]),
        ("search_issues", "waitlist state:open", [52]),
        ("search_issues", "is:issue label:bug label:billing", [43]),
        ("search_issues", "Spotify assignee:hossein-dropsubs", [13]),
        ("search_issues", "Spotify author:danial-dropsubs", [36]),
        ("search_issues", "Gmail is:pr", []),
        ("search_pull_requests", "waitlist is:merged", [17]),
        ("search_pull_requests", "waitlist is:unmerged", [53]),
        ("search_pull_requests", "is:pr is:open", [54, 53]),
        ("search_pull_requests", "is:draft", [54]),
    ],
)
async def test_search_qualifiers(tool, query, found):
    assert await search(tool, query, owner=OWNER, repo=REPO) == found


@pytest.mark.parametrize(
    "query",
    [
        "",
        "Gmail language:python",
        "Gmail created:>2026-01-01",
        "Gmail in:title",
        "Gmail OR Outlook",
        "NOT Gmail",
        "Gmail -label:bug",
        "Gmail is:locked",
        'Gmail "unclosed',
    ],
)
async def test_unsupported_search_syntax_is_a_clear_error_not_everything(query):
    result = await call("search_issues", query=query, owner=OWNER, repo=REPO)

    assert result.is_error
    assert text(result)


async def test_searches_sort_and_page():
    oldest = await search("search_issues", "fee", sort="created", order="asc")
    most_discussed = await search("search_issues", "fee", sort="comments", order="desc")
    page_two = await search("search_issues", "fee", sort="created", order="asc", perPage=2, page=2)

    assert oldest == [11, 20, 23, 37, 43, 47]
    assert most_discussed[0] == 37  # 8 comments
    assert page_two == oldest[2:4]
    data = payload(await call("search_issues", query="fee", perPage=2))
    assert data["total_count"] == 6 and len(data["items"]) == 2


@pytest.mark.parametrize(
    "arguments", [{"sort": "reactions"}, {"perPage": 0}, {"perPage": 101}, {"page": 0}]
)
async def test_unsupported_sort_or_paging_is_an_error(arguments):
    assert (await call("search_issues", query="fee", **arguments)).is_error


# list_issues and issue_read


async def test_list_issues_lists_every_issue_newest_first_in_the_minimal_graphql_shape():
    data = await read("list_issues")

    expected = sorted(raw("issues"), key=lambda i: i["created_at"], reverse=True)
    assert numbers(data["issues"]) == numbers(expected)
    assert data["totalCount"] == 30
    assert data["pageInfo"]["hasNextPage"] is False
    first = data["issues"][0]
    assert first["state"] in ("OPEN", "CLOSED")
    assert first["user"]["login"]
    assert all(isinstance(label, str) for label in first["labels"])
    assert isinstance(first["assignees"], list)


async def test_list_issues_filters_and_orders():
    open_issues = await read("list_issues", state="OPEN")
    bugs = await read("list_issues", labels=["bug"])
    since = await read("list_issues", since="2026-10-02")
    by_update = await read("list_issues", orderBy="UPDATED_AT", direction="ASC")

    assert sorted(numbers(open_issues["issues"])) == [25, 29, 31, 36, 37, 38, 41, 45, 47, 51, 52]
    assert {i["state"] for i in open_issues["issues"]} == {"OPEN"}
    assert sorted(numbers(bugs["issues"])) == [10, 14, 16, 32, 36, 42, 43, 49]
    assert sorted(numbers(since["issues"])) == [29, 42, 43, 49, 51, 52]
    updated = [i["updated_at"] for i in by_update["issues"]]
    assert updated == sorted(updated)


async def test_list_issues_pages_by_cursor():
    everything = numbers((await read("list_issues"))["issues"])

    first = await read("list_issues", perPage=5)
    second = await read("list_issues", perPage=5, after=first["pageInfo"]["endCursor"])

    assert numbers(first["issues"]) == everything[:5]
    assert first["pageInfo"]["hasNextPage"] is True
    assert numbers(second["issues"]) == everything[5:10]
    assert second["pageInfo"]["hasPreviousPage"] is True


@pytest.mark.parametrize(
    "arguments",
    [
        {"page": 2},
        {"after": "not-a-cursor"},
        {"since": "last week"},
        {"field_filters": [{"field_name": "Priority", "value": "P1"}]},
        {"perPage": 101},
    ],
)
async def test_list_issues_refuses_what_it_does_not_support(arguments):
    assert (await call("list_issues", owner=OWNER, repo=REPO, **arguments)).is_error


async def test_an_issue_reads_in_the_minimal_shape_with_the_prs_that_close_it():
    issue = await read("issue_read", method="get", issue_number=43)

    (expected,) = [i for i in raw("issues") if i["number"] == 43]
    assert issue["number"] == 43 and issue["title"] == expected["title"]
    assert issue["body"] == expected["body"]
    assert issue["state"] == "closed" and issue["state_reason"] == "completed"
    assert issue["html_url"] == f"{WEB}/issues/43"
    assert issue["user"]["login"] == expected["user"]["login"]
    assert issue["labels"] == ["bug", "billing", "backend"]
    assert issue["assignees"] == ["mohammadreza-dropsubs"]
    assert issue["comments"] == 2
    assert issue["closed_at"] == "2026-10-01T18:00:00Z"
    closing = issue["closed_by_pull_requests"]
    assert closing["total_count"] == 1
    assert closing["references"] == [
        {
            "number": 46,
            "title": "Inbox sync: check for a repeated page token before appending",
            "state": "MERGED",
            "url": f"{WEB}/pull/46",
            "repository": FULL_NAME,
        }
    ]


async def test_issue_comments_and_labels():
    comments = await read("issue_read", method="get_comments", issue_number=37)
    page = await read("issue_read", method="get_comments", issue_number=37, perPage=3, page=2)
    labels = await read("issue_read", method="get_labels", issue_number=43)

    expected = [c for c in raw("comments") if c["issue_number"] == 37]
    assert [c["id"] for c in comments] == [c["id"] for c in expected]
    assert len(comments) == 8
    assert comments[0]["body"] == expected[0]["body"]
    assert comments[0]["html_url"] == expected[0]["html_url"]
    assert comments[0]["user"]["login"] == expected[0]["user"]["login"]
    assert [c["id"] for c in page] == [c["id"] for c in expected[3:6]]
    assert [label["name"] for label in labels["labels"]] == ["bug", "billing", "backend"]
    assert labels["totalCount"] == 3


@pytest.mark.parametrize(
    "arguments",
    [
        {"method": "get", "issue_number": 999},
        {"method": "get_comments", "issue_number": 999},
        {"method": "get_sub_issues", "issue_number": 43},
        {"method": "get_parent", "issue_number": 43},
    ],
)
async def test_issue_reads_of_missing_issues_or_data_are_errors(arguments):
    assert (await call("issue_read", owner=OWNER, repo=REPO, **arguments)).is_error


# list_pull_requests and pull_request_read


async def test_list_pull_requests_lists_open_ones_by_default_in_the_minimal_shape():
    prs = await read("list_pull_requests")

    assert numbers(prs) == [54, 53]
    draft = prs[0]
    assert draft["draft"] is True and draft["merged"] is False
    assert draft["state"] == "open"
    assert draft["html_url"] == f"{WEB}/pull/54"
    assert draft["head"]["ref"] == "fe/delete-account"
    assert draft["base"]["ref"] == "main"
    assert draft["user"]["login"] == "reyhaneh-dropsubs"
    assert "merged_at" not in draft


async def test_list_pull_requests_filters_sorts_and_pages():
    closed = await read("list_pull_requests", state="closed", perPage=100)
    by_update = await read("list_pull_requests", state="all", sort="updated", direction="asc")
    page = await read("list_pull_requests", state="all", perPage=5, page=2)
    everything = await read("list_pull_requests", state="all", perPage=100)
    by_head = await read("list_pull_requests", state="all", head=f"{OWNER}:fe/delete-account")
    other_base = await read("list_pull_requests", state="all", base="release")
    trimmed = await read("list_pull_requests", state="closed", fields=["number", "merged_at"])

    assert len(closed) == 22 and all(pr["merged"] for pr in closed)
    assert closed[0]["number"] == 50 and closed[0]["merged_at"] == "2026-10-02T23:40:00Z"
    updated = [pr["updated_at"] for pr in by_update]
    assert updated == sorted(updated)
    assert numbers(page) == numbers(everything)[5:10]
    assert numbers(by_head) == [54]
    assert other_base == []
    assert trimmed[0] == {"number": 50, "merged_at": "2026-10-02T23:40:00Z"}


async def test_a_merged_pull_request_reads_with_its_merge():
    pr = await read("pull_request_read", method="get", pullNumber=50)

    assert pr["number"] == 50
    assert pr["title"] == "Enforce the Approve check inside the cancel service"
    assert pr["state"] == "closed"
    assert pr["merged"] is True
    assert pr["merged_at"] == "2026-10-02T23:40:00Z"
    assert pr["merged_by"] == "hossein-dropsubs"
    assert pr["html_url"] == f"{WEB}/pull/50"
    assert pr["labels"] == ["security", "ai"]
    assert pr["changed_files"] == 2


async def test_an_open_pull_request_is_not_merged():
    pr = await read("pull_request_read", method="get", pullNumber=53)

    assert pr["merged"] is False and pr["state"] == "open"
    assert "merged_at" not in pr


async def test_pull_request_status_files_commits_and_comments():
    status = await read("pull_request_read", method="get_status", pullNumber=50)
    files = await read("pull_request_read", method="get_files", pullNumber=2)
    commits = await read("pull_request_read", method="get_commits", pullNumber=50)
    comments = await read("pull_request_read", method="get_comments", pullNumber=50)

    (expected,) = [p for p in raw("pull_requests") if p["number"] == 50]
    assert status["state"] == "success" and status["total_count"] == 2
    assert status["sha"] == expected["head"]["sha"]
    assert {s["context"] for s in status["statuses"]} == {"ci/tests", "ci/lint"}
    assert [f["filename"] for f in files] == ["api/sync/gmail.py", "api/sync/test_gmail.py"]
    assert files[0]["status"] == "added" and files[0]["additions"] == 212
    assert [c["sha"] for c in commits] == [head("demo")]
    assert commits[0]["message"].startswith("Enforce the Approve check")
    assert comments == []


@pytest.mark.parametrize(
    "arguments",
    [
        {"method": "get", "pullNumber": 43},  # an issue, not a pull request
        {"method": "get", "pullNumber": 999},
        {"method": "get_diff", "pullNumber": 50},
        {"method": "get_reviews", "pullNumber": 50},
        {"method": "get_review_comments", "pullNumber": 50},
        {"method": "get_check_runs", "pullNumber": 50},
    ],
)
async def test_pull_request_reads_of_missing_prs_or_data_are_errors(arguments):
    assert (await call("pull_request_read", owner=OWNER, repo=REPO, **arguments)).is_error


@pytest.mark.parametrize(
    "arguments", [{"sort": "long-running"}, {"perPage": 0}, {"state": "merged"}]
)
async def test_list_pull_requests_refuses_what_it_does_not_support(arguments):
    assert (await call("list_pull_requests", owner=OWNER, repo=REPO, **arguments)).is_error


# list_commits


async def test_commits_list_the_default_branch_newest_first_in_the_minimal_shape():
    commits = await read("list_commits")

    assert [c["sha"] for c in commits] == [c["sha"] for c in raw("commits")]
    first = commits[0]
    assert first["sha"] == head("demo")
    assert first["html_url"] == f"{WEB}/commit/{head('demo')}"
    assert first["commit"]["message"].startswith("Enforce the Approve check")
    assert first["commit"]["author"]["date"] == "2026-10-02T23:40:00Z"
    assert first["author"]["login"] == "hossein-dropsubs"


async def test_the_dev_snapshots_main_ends_at_its_own_head(monkeypatch):
    monkeypatch.setenv("WORLD_SNAPSHOT", "dev")

    commits = await read("list_commits")

    assert commits[0]["sha"] == head("dev")
    assert commits[0]["commit"]["message"].startswith("Billing: 30% success fee")


async def test_commits_filter_and_page():
    shas = [c["sha"] for c in raw("commits")]
    at_tag = await read("list_commits", sha="v0.9.3")
    at_short = await read("list_commits", sha=shas[4][:7])
    by_login = await read("list_commits", author="reyhaneh-dropsubs")
    by_email = await read("list_commits", author="reyhaneh@dropsubs.example")
    by_path = await read("list_commits", path="api/billing")
    window = await read("list_commits", since="2026-10-01", until="2026-10-02T00:00:00Z")
    page = await read("list_commits", perPage=5, page=2)
    trimmed = await read("list_commits", perPage=1, fields=["sha"])

    assert at_tag[0]["sha"].startswith("65e249e6")
    assert [c["sha"] for c in at_short] == shas[4:]
    assert by_login and {c["author"]["login"] for c in by_login} == {"reyhaneh-dropsubs"}
    assert [c["sha"] for c in by_email] == [c["sha"] for c in by_login]
    assert [c["commit"]["message"] for c in by_path] == [
        "Billing: migrate to payment provider API v2 (#26)",
        "Billing: 30% success fee (#22)",
    ]
    assert [c["sha"][:8] for c in window] == ["e20e7e57", "83f12fb0"]
    assert [c["sha"] for c in page] == shas[5:10]
    assert trimmed == [{"sha": shas[0]}]


@pytest.mark.parametrize(
    "arguments", [{"sha": "no-such-branch"}, {"since": "yesterday"}, {"perPage": 500}]
)
async def test_list_commits_refuses_unknown_refs_and_bad_arguments(arguments):
    assert (await call("list_commits", owner=OWNER, repo=REPO, **arguments)).is_error


# list_releases and get_latest_release


async def test_releases_list_newest_first_in_the_minimal_shape():
    releases = await read("list_releases")

    assert [r["tag_name"] for r in releases] == [
        "v0.9.3",
        "v0.9.1",
        "v0.8.0",
        "v0.7.0",
        "v0.6.0",
        "v0.5.0",
        "v0.4.1",
        "v0.4.0",
        "v0.3.0",
    ]
    latest = releases[0]
    assert latest["published_at"] == "2026-09-30T22:00:00Z"
    assert latest["html_url"] == f"{WEB}/releases/tag/v0.9.3"
    assert latest["draft"] is False and latest["prerelease"] is False
    assert latest["body"].startswith("AI reads full email bodies.")
    assert latest["author"]["login"] == "mohammadreza-dropsubs"
    assert not any(key.startswith("_") for key in latest)


async def test_releases_page_and_trim():
    first_two = await read("list_releases", perPage=2)
    next_two = await read("list_releases", perPage=2, page=2)
    trimmed = await read("list_releases", perPage=1, fields=["tag_name", "published_at"])

    assert [r["tag_name"] for r in first_two] == ["v0.9.3", "v0.9.1"]
    assert [r["tag_name"] for r in next_two] == ["v0.8.0", "v0.7.0"]
    assert trimmed == [{"tag_name": "v0.9.3", "published_at": "2026-09-30T22:00:00Z"}]


async def test_the_latest_release_comes_back_as_the_rest_release():
    release = await read("get_latest_release")

    (expected,) = [r for r in raw("releases") if r["tag_name"] == "v0.9.3"]
    assert release == {k: v for k, v in expected.items() if not k.startswith("_")}


async def test_pull_requests_merged_after_the_latest_release_exist():
    """The fact-checks rely on these: merged, but not in any release yet."""
    latest = at((await read("get_latest_release"))["published_at"])
    merged = await read("list_pull_requests", state="closed", perPage=100)

    later = [pr["number"] for pr in merged if at(pr["merged_at"]) > latest]
    assert later == [50, 48, 46]


# Every read is of the world's repository


@pytest.mark.parametrize(
    ("tool", "arguments"),
    [
        ("list_issues", {}),
        ("issue_read", {"method": "get", "issue_number": 43}),
        ("list_pull_requests", {}),
        ("pull_request_read", {"method": "get", "pullNumber": 50}),
        ("list_commits", {}),
        ("list_releases", {}),
        ("get_latest_release", {}),
        ("add_issue_comment", {"issue_number": 43, "body": "x"}),
    ],
)
async def test_another_repository_is_not_found(tool, arguments, overlay_dir):
    result = await call(tool, owner="otherorg", repo="private-repo", **arguments)

    assert result.is_error
    assert "Not Found" in text(result)
    assert JournalOverlay(overlay_dir, "demo").journal() == []


async def test_the_owner_and_repo_ignore_case():
    issue = payload(
        await call("issue_read", method="get", owner=OWNER.upper(), repo=REPO, issue_number=43)
    )

    assert issue["number"] == 43


# search_code stays strict too


@pytest.mark.parametrize(
    "query",
    [
        f"FEE_RATE language:python repo:{FULL_NAME}",
        f"FEE_RATE OR REFUND repo:{FULL_NAME}",
        f"NOT FEE_RATE repo:{FULL_NAME}",
    ],
)
async def test_code_search_refuses_unsupported_syntax(query):
    assert (await call("search_code", query=query)).is_error


async def test_code_search_still_finds_files():
    data = payload(await call("search_code", query=f"FEE_RATE repo:{FULL_NAME}"))

    assert [item["path"] for item in data["items"]] == ["api/billing/fees.py"]


# add_issue_comment goes to the overlay


async def test_a_comment_is_journaled_and_reads_back(overlay_dir):
    before = json.loads((DATA / "comments.json").read_text())

    created = payload(
        await call(
            "add_issue_comment",
            owner=OWNER,
            repo=REPO,
            issue_number=36,
            body="Zanzibar moved to 6am.",
        )
    )

    comments = await read("issue_read", method="get_comments", issue_number=36)
    issue = await read("issue_read", method="get", issue_number=36)
    assert comments[-1]["body"] == "Zanzibar moved to 6am."
    assert str(comments[-1]["id"]) == created["id"]
    assert created["url"] == comments[-1]["html_url"]
    assert comments[-1]["html_url"].startswith(f"{WEB}/issues/36#issuecomment-")
    assert issue["comments"] == 4
    (entry,) = JournalOverlay(overlay_dir, "demo").journal()
    assert (entry.server, entry.tool, entry.op, entry.target) == (
        "github",
        "add_issue_comment",
        "add_comment",
        "36",
    )
    assert json.loads((DATA / "comments.json").read_text()) == before
    assert await search("search_issues", "zanzibar", owner=OWNER, repo=REPO) == [36]


async def test_a_comment_on_a_pull_request_reads_back_with_it():
    await call("add_issue_comment", owner=OWNER, repo=REPO, issue_number=53, body="Hold this.")

    comments = await read("pull_request_read", method="get_comments", pullNumber=53)

    assert [c["body"] for c in comments] == ["Hold this."]


@pytest.mark.parametrize(
    "arguments", [{"issue_number": 999, "body": "x"}, {"issue_number": 36, "body": "  "}]
)
async def test_an_invalid_comment_is_refused_and_nothing_is_written(arguments, overlay_dir):
    result = await call("add_issue_comment", owner=OWNER, repo=REPO, **arguments)

    assert result.is_error
    assert JournalOverlay(overlay_dir, "demo").journal() == []


async def test_reset_forgets_comments():
    await call("add_issue_comment", owner=OWNER, repo=REPO, issue_number=36, body="Temporary")

    overlay.main()

    comments = await read("issue_read", method="get_comments", issue_number=36)
    assert "Temporary" not in [c["body"] for c in comments]


def test_the_loader_returns_copies_of_the_github_data():
    pulls = mock_records("github", "pull_requests")
    pulls[0]["title"] = "changed"

    assert mock_records("github", "pull_requests") == raw("pull_requests")
