"""The mock GitHub server's code tools answer from the snapshot's repository files, in the shapes
of GitHub's official MCP server."""

import json

import pytest
from mcp import Client
from mcp.types import EmbeddedResource, TextContent, TextResourceContents

from world import code
from world.config import SNAPSHOTS_DIR, world_spec
from world.github_mcp import server

pytestmark = pytest.mark.anyio

OWNER, REPO = world_spec().github_repo.split("/")
FEES = "api/billing/fees.py"


@pytest.fixture(params=["dev", "demo"])
def snapshot(request, monkeypatch) -> str:
    monkeypatch.setenv("WORLD_SNAPSHOT", request.param)
    return request.param


def head(snapshot: str) -> str:
    return json.loads((SNAPSHOTS_DIR / snapshot / "repo.json").read_text())["sha"]


def on_disk(snapshot: str, path: str) -> str:
    return (SNAPSHOTS_DIR / snapshot / "repo" / path).read_text()


async def call(tool: str, **arguments):
    async with Client(server) as client:
        return await client.call_tool(tool, arguments)


def payload(result) -> dict:
    if result.structured_content is not None:
        return result.structured_content
    return json.loads("".join(c.text for c in result.content if isinstance(c, TextContent)))


async def test_code_search_finds_the_repos_files_in_the_minimal_shape(snapshot):
    result = await call("search_code", query=f"FEE_RATE repo:{OWNER}/{REPO}")

    data = payload(result)
    assert data["total_count"] == len(data["items"]) == 1
    (item,) = data["items"]
    assert item["path"] == FEES and item["name"] == "fees.py"
    assert item["repository"] == f"{OWNER}/{REPO}"
    assert len(item["sha"]) == 40


async def test_code_search_needs_every_word(snapshot):
    data = payload(await call("search_code", query=f"FEE_RATE nonexistentword repo:{OWNER}/{REPO}"))

    assert data["items"] == []


async def test_code_search_scoped_elsewhere_finds_nothing(snapshot):
    data = payload(await call("search_code", query="FEE_RATE repo:otherorg/private-repo"))

    assert data["items"] == []


async def test_a_file_comes_back_as_text_with_its_commit_in_the_uri(snapshot):
    result = await call("get_file_contents", owner=OWNER, repo=REPO, path=FEES, ref="main")

    assert not result.is_error
    message, resource = result.content
    assert isinstance(message, TextContent) and message.text.startswith(
        "successfully downloaded text file (SHA: "
    )
    assert isinstance(resource, EmbeddedResource)
    assert isinstance(resource.resource, TextResourceContents)
    assert resource.resource.text == on_disk(snapshot, FEES)
    assert (
        str(resource.resource.uri) == f"repo://{OWNER}/{REPO}/sha/{head(snapshot)}/contents/{FEES}"
    )


def test_snapshots_hold_the_code_of_their_date():
    assert '"0 0 * * *"' in on_disk("dev", "api/jobs/schedule.py")
    assert '"0 6 * * *"' in on_disk("demo", "api/jobs/schedule.py")
    assert not (SNAPSHOTS_DIR / "dev" / "repo" / "api/cancel/service.py").exists()


async def test_the_default_branch_and_its_commit_read_the_same_file(snapshot):
    by_default = await call("get_file_contents", owner=OWNER, repo=REPO, path=FEES)
    by_sha = await call("get_file_contents", owner=OWNER, repo=REPO, path=FEES, sha=head(snapshot))
    by_ref = await call(
        "get_file_contents", owner=OWNER, repo=REPO, path=FEES, ref="refs/heads/main"
    )

    texts = {r.content[1].resource.text for r in (by_default, by_sha, by_ref)}
    assert texts == {on_disk(snapshot, FEES)}


@pytest.mark.parametrize(
    "arguments",
    [
        {"owner": OWNER, "repo": REPO, "path": "../../world.json"},
        {"owner": OWNER, "repo": REPO, "path": "api/../../repo.json"},
        {"owner": OWNER, "repo": REPO, "path": "api/nope.py"},
        {"owner": OWNER, "repo": REPO, "path": FEES, "ref": "no-such-branch"},
        {"owner": OWNER, "repo": REPO, "path": FEES, "sha": "0" * 40},
        {"owner": "otherorg", "repo": "private-repo", "path": FEES},
    ],
)
async def test_reads_outside_the_repo_or_snapshot_fail(snapshot, arguments):
    result = await call("get_file_contents", **arguments)

    assert result.is_error


async def test_a_directory_lists_its_entries(snapshot):
    result = await call("get_file_contents", owner=OWNER, repo=REPO, path="api/billing")

    entries = payload(result)
    assert any(e["path"] == FEES and e["type"] == "file" for e in entries)


def test_the_repo_loader_reads_every_file():
    tree = code.load_repo("demo")

    assert tree.sha == head("demo")
    assert tree.files[FEES] == on_disk("demo", FEES)
    assert "api/cancel/service.py" in tree.files
