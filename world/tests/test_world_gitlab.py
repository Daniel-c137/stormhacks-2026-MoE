"""The mock GitLab server answers with the tool names and arguments of GitLab's official MCP
server, over mock-data/gitlab/ and the snapshot's code, and refuses what it cannot answer."""

import json

import pytest
from mcp import Client
from mcp.types import TextContent

from world import gitlab_mcp
from world.code import load_tree
from world.config import MOCK_DATA_DIR, SNAPSHOTS_DIR, world_spec
from world.gitlab_mcp import server

pytestmark = pytest.mark.anyio

PROJECT = "dropsubs/infra"
WEB = f"https://gitlab.com/{PROJECT}"
DATA = MOCK_DATA_DIR / "gitlab" / PROJECT


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(autouse=True)
def demo(monkeypatch):
    monkeypatch.setenv("WORLD_SNAPSHOT", "demo")


async def call(tool: str, **arguments):
    async with Client(server) as client:
        return await client.call_tool(tool, arguments)


def text(result) -> str:
    return "".join(c.text for c in result.content if isinstance(c, TextContent))


def payload(result):
    assert not result.is_error, text(result)
    return json.loads(text(result))


def raw(kind: str):
    return json.loads((DATA / f"{kind}.json").read_text())


def test_the_world_has_a_gitlab_project_with_its_own_records_and_code():
    assert world_spec().gitlab_projects == [PROJECT]
    for kind in ("issues", "merge_requests"):
        for record in raw(kind):
            assert record["web_url"].startswith(f"{WEB}/-/"), record["web_url"]
    head = json.loads((SNAPSHOTS_DIR / "demo/repos/gitlab/dropsubs/infra.json").read_text())
    merged = [m["merge_commit_sha"] for m in raw("merge_requests") if m["state"] == "merged"]
    assert head["sha"] in merged
    assert "monitoring/status.yml" in load_tree("demo", "gitlab", PROJECT).files
    assert "monitoring/status.yml" not in load_tree("dev", "gitlab", PROJECT).files


async def test_the_server_offers_the_official_read_tools_and_no_writes():
    async with Client(server) as client:
        tools = {tool.name for tool in (await client.list_tools()).tools}

    assert {
        "search",
        "get_work_item",
        "list_work_items",
        "get_merge_request",
        "list_merge_requests",
        "get_repository_file",
        "list_releases",
    } <= tools
    assert not tools & {"save_work_item", "save_merge_request", "save_note", "create_issue"}


async def test_search_finds_merge_requests_and_issues_of_the_project():
    mrs = payload(
        await call("search", scope="merge_requests", search="status page", project_id=PROJECT)
    )
    issues = payload(await call("search", scope="issues", search="load test", project_id=PROJECT))

    assert [m["iid"] for m in mrs] == [4]
    assert mrs[0]["state"] == "merged" and mrs[0]["web_url"] == f"{WEB}/-/merge_requests/4"
    assert mrs[0]["references"]["full"] == f"{PROJECT}!4"
    assert "_diffs" not in mrs[0]
    assert [i["iid"] for i in issues] == [6, 5]  # newest first; #6 waits on the load test
    assert issues[0]["state"] == "opened"


async def test_search_filters_by_state_and_needs_every_word():
    merged = payload(
        await call(
            "search", scope="merge_requests", search="load", state="merged", project_id=PROJECT
        )
    )
    nothing = payload(await call("search", scope="issues", search="load nonexistentword"))

    assert merged == []  # the load test MR is still open
    assert nothing == []


@pytest.mark.parametrize("scope", ["commits", "wiki_pages", "everything"])
async def test_a_scope_the_data_cannot_answer_is_an_error(scope):
    assert (await call("search", scope=scope, search="deploy", project_id=PROJECT)).is_error


async def test_code_search_returns_blobs_with_the_matching_lines():
    hits = payload(
        await call("search", scope="blobs", search="interval_seconds", project_id=PROJECT)
    )

    (hit,) = hits
    assert hit["path"] == "monitoring/status.yml"
    assert "interval_seconds: 60" in hit["data"]
    assert hit["project_id"] == raw("project")["id"]


async def test_a_work_item_reads_in_the_graphql_shape_with_its_notes():
    item = payload(
        await call("get_work_item", project_id=PROJECT, work_item_iid=5, include=["notes"])
    )

    assert (item["iid"], item["state"], item["webUrl"]) == ("5", "OPEN", f"{WEB}/-/issues/5")
    assert item["workItemType"]["name"] == "Issue"
    assert [n["body"] for n in item["notes"]] == [
        "Not before Sunday: the refunds and the App Store review come first."
    ]


async def test_a_work_item_reads_by_url_too():
    item = payload(await call("get_work_item", url=f"{WEB}/-/issues/3"))

    assert item["title"] == "Status page for the API and the sync workers"
    assert item["state"] == "CLOSED"


async def test_a_merge_request_reads_with_its_diffs_at_each_detail():
    plain = payload(await call("get_merge_request", project_id=PROJECT, merge_request_iid=4))
    stats = payload(
        await call("get_merge_request", project_id=PROJECT, merge_request_iid=4, include=["diffs"])
    )
    full = payload(
        await call(
            "get_merge_request",
            project_id=PROJECT,
            merge_request_iid=4,
            include=["diffs"],
            detail="full_patch",
        )
    )

    assert plain["state"] == "merged" and plain["merged_at"] == "2026-09-28T23:00:00Z"
    assert "diffs" not in plain
    assert stats["diffs"][0]["new_path"] == "monitoring/status.yml"
    assert "diff" not in stats["diffs"][0] and stats["diffs"][0]["additions"] > 0
    assert "+interval_seconds: 60" in full["diffs"][0]["diff"]


@pytest.mark.parametrize(
    "arguments",
    [
        {"merge_request_iid": 99},
        {"merge_request_iid": 4, "include": ["diffs", "notes"]},
        {"merge_request_iid": 4, "include": ["pipelines"]},
    ],
)
async def test_merge_request_reads_the_data_cannot_answer_are_errors(arguments):
    assert (await call("get_merge_request", project_id=PROJECT, **arguments)).is_error


async def test_merge_requests_list_newest_first_and_filter():
    every = payload(await call("list_merge_requests", project_id=PROJECT))
    opened = payload(await call("list_merge_requests", project_id=PROJECT, state="opened"))

    assert [m["iid"] for m in every["items"]] == [7, 4, 2]
    assert [m["reference"] for m in opened["items"]] == [f"{PROJECT}!7"]


async def test_work_items_list_and_page_by_cursor():
    first = payload(await call("list_work_items", project_id=PROJECT, first=2))
    rest = payload(
        await call(
            "list_work_items", project_id=PROJECT, first=2, after=first["pageInfo"]["endCursor"]
        )
    )

    assert [i["iid"] for i in first["items"]] == ["6", "5"]
    assert first["pageInfo"]["hasNextPage"] is True
    assert [i["iid"] for i in rest["items"]] == ["3", "1"]


async def test_a_file_reads_at_head_with_its_commit_and_line_window():
    head = load_tree("demo", "gitlab", PROJECT).sha
    whole = payload(
        await call(
            "get_repository_file", project_id=PROJECT, file_path="monitoring/status.yml", ref="HEAD"
        )
    )
    window = payload(
        await call(
            "get_repository_file",
            project_id=PROJECT,
            file_path="monitoring/status.yml",
            ref="main",
            offset=1,
            limit=2,
        )
    )

    assert whole["commit_id"] == head
    assert whole["content"].startswith("# The status page")
    assert whole["truncated"] is False
    assert window["content"].split("\n") == ["interval_seconds: 60", "checks:"]
    assert window["truncated"] is True


@pytest.mark.parametrize(
    "arguments",
    [
        {"file_path": "missing.yml", "ref": "HEAD"},
        {"file_path": "../secrets", "ref": "HEAD"},
        {"file_path": "monitoring/status.yml", "ref": "some-branch"},
    ],
)
async def test_file_reads_outside_the_tree_or_ref_fail(arguments):
    assert (await call("get_repository_file", project_id=PROJECT, **arguments)).is_error


@pytest.mark.parametrize(
    ("tool", "arguments"),
    [
        ("get_work_item", {"work_item_iid": 5}),
        ("get_merge_request", {"merge_request_iid": 4}),
        ("get_repository_file", {"file_path": "monitoring/status.yml", "ref": "HEAD"}),
        ("list_releases", {}),
    ],
)
async def test_another_project_is_not_found(tool, arguments):
    result = await call(tool, project_id="someone/else", **arguments)

    assert result.is_error
    assert "Not Found" in text(result)


async def test_the_project_has_no_releases_and_says_so_with_an_empty_list():
    assert payload(await call("list_releases", project_id=PROJECT)) == []


def test_main_listens_on_the_configured_host_and_port(monkeypatch):
    calls = []
    monkeypatch.setattr(gitlab_mcp.server, "run", lambda *a, **k: calls.append((a, k)))
    monkeypatch.setenv("WORLD_MCP_HOST", "0.0.0.0")
    monkeypatch.setenv("WORLD_GITLAB_MCP_PORT", "9103")

    gitlab_mcp.main()

    assert calls == [(("streamable-http",), {"host": "0.0.0.0", "port": 9103})]
