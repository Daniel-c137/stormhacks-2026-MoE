"""Code search and file reads in the team's repository, and snippets copied from the file: the
code is always the file's own lines, and the link pins the commit it came from."""

import pytest
from conftest import FEES_PY, MAIN_SHA, REFUNDS_PY, RELEASE_SHA

from brain.agent.code import MAX_SNIPPET_LINES, code_evidence, language_for, snippet_from_file
from brain.github import CodeFile, GitHubError, GitHubReader, UnsafePath
from brain.integrations import READ_TOOLS

pytestmark = pytest.mark.anyio

REFUNDS = "api/billing/refunds.py"


def reader(fake_github, ref: str | None = None) -> GitHubReader:
    return GitHubReader("dropsubs/app", fake_github.server, ref=ref)


def lines(text: str, start: int, end: int) -> str:
    return "\n".join(text.splitlines()[start - 1 : end])


def code_file(text: str, path: str = REFUNDS, ref: str = MAIN_SHA, pinned=True) -> CodeFile:
    return CodeFile(repo="dropsubs/app", path=path, ref=ref, pinned=pinned, text=text)


def test_the_official_servers_code_tools_are_read_tools():
    assert {"search_code", "get_file_contents"} <= READ_TOOLS["github"]


# search


async def test_code_search_is_scoped_to_the_team_repo(fake_github):
    hits = await reader(fake_github).search_code("refund window")

    assert [h.path for h in hits] == [REFUNDS]
    ((tool, args),) = fake_github.calls
    assert tool == "search_code"
    assert args["query"] == "refund window repo:dropsubs/app"


async def test_qualifiers_in_the_search_text_cannot_widen_the_scope(fake_github):
    hits = await reader(fake_github).search_code(
        "REFUND_WINDOW_DAYS repo:otherorg/private-repo org:otherorg OR secret NOT x path:/"
    )

    ((_, args),) = fake_github.calls
    query = args["query"]
    assert query.endswith(" repo:dropsubs/app")
    assert query.count(":") == 1
    assert "repo:otherorg" not in query and "org:" not in query and "path:" not in query
    assert " OR " not in f" {query} " and " NOT " not in f" {query} "
    assert all(h.path != "refunds_secret.py" for h in hits)


async def test_code_results_outside_the_team_repo_are_dropped(fake_github):
    fake_github.ignore_scope = True

    hits = await reader(fake_github).search_code("REFUND_WINDOW_DAYS")

    assert [h.path for h in hits] == [REFUNDS]


async def test_search_text_without_words_makes_no_call(fake_github):
    assert await reader(fake_github).search_code("  :: ** ") == []
    assert fake_github.calls == []


# file reads


async def test_a_file_read_pins_the_commit_the_server_resolved(fake_github):
    file = await reader(fake_github).read_file(REFUNDS)

    assert file.text == REFUNDS_PY
    assert (file.ref, file.pinned) == (MAIN_SHA, True)
    ((tool, args),) = fake_github.calls
    assert tool == "get_file_contents"
    assert (args["owner"], args["repo"], args["path"]) == ("dropsubs", "app", REFUNDS)


async def test_a_file_read_uses_the_teams_ref(fake_github):
    file = await reader(fake_github, ref="release").read_file(REFUNDS)

    assert "REFUND_WINDOW_DAYS = 14" in file.text
    assert file.ref == RELEASE_SHA
    ((_, args),) = fake_github.calls
    assert args["ref"] == "release"


async def test_without_a_resolved_commit_the_link_falls_back_to_the_branch(fake_github):
    fake_github.resolve_refs = False

    file = await reader(fake_github, ref="release").read_file(REFUNDS)

    assert (file.ref, file.pinned) == ("release", False)


@pytest.mark.parametrize(
    "path",
    [
        "../secrets.py",
        "api/../../etc/passwd",
        "api/./../x.py",
        "/etc/passwd",
        "https://github.com/otherorg/private-repo/blob/main/x.py",
        "file:///etc/passwd",
        "api\\..\\x.py",
        "",
        "api/billing/",
    ],
)
async def test_unsafe_paths_are_rejected_before_any_call(fake_github, path):
    with pytest.raises(UnsafePath):
        await reader(fake_github).read_file(path)
    assert fake_github.calls == []


async def test_a_missing_file_is_an_error_not_an_empty_file(fake_github):
    with pytest.raises(GitHubError):
        await reader(fake_github).read_file("api/nope.py")


# snippets


def test_a_snippet_is_the_files_own_lines():
    snippet = snippet_from_file(code_file(REFUNDS_PY), 6, 7, caption="The refund window")

    assert snippet.code == lines(REFUNDS_PY, 6, 7)
    assert snippet.code.splitlines()[0] == "REFUND_WINDOW_DAYS = 30"
    assert (snippet.path, snippet.start_line, snippet.end_line) == (REFUNDS, 6, 7)
    assert snippet.language == "python"
    assert snippet.caption == "The refund window"


def test_the_link_is_a_permalink_to_the_commit_and_lines():
    snippet = snippet_from_file(code_file(REFUNDS_PY), 6, 7)

    assert snippet.github_url == (
        f"https://github.com/dropsubs/app/blob/{MAIN_SHA}/api/billing/refunds.py#L6-L7"
    )


def test_an_unpinned_link_names_the_branch():
    snippet = snippet_from_file(code_file(REFUNDS_PY, ref="main", pinned=False), 6, 6)

    assert (
        snippet.github_url
        == "https://github.com/dropsubs/app/blob/main/api/billing/refunds.py#L6-L6"
    )


def test_an_invented_range_past_the_end_is_clamped_to_the_file():
    total = len(REFUNDS_PY.splitlines())

    snippet = snippet_from_file(code_file(REFUNDS_PY), 9, 400)

    assert (snippet.start_line, snippet.end_line) == (9, total)
    assert snippet.code == lines(REFUNDS_PY, 9, total)
    assert snippet.github_url.endswith(f"#L9-L{total}")


@pytest.mark.parametrize("start, end", [(200, 210), (0, 0), (-5, -1), (8, 3)])
def test_a_range_outside_the_file_is_rejected(start, end):
    with pytest.raises(ValueError):
        snippet_from_file(code_file(REFUNDS_PY), start, end)


def test_a_snippet_is_capped():
    text = "".join(f"line_{i} = {i}\n" for i in range(1, 201))

    snippet = snippet_from_file(code_file(text, path="api/big.py"), 10, 190)

    assert (snippet.start_line, snippet.end_line) == (10, 10 + MAX_SNIPPET_LINES - 1)
    assert snippet.code == lines(text, 10, 10 + MAX_SNIPPET_LINES - 1)


def test_the_highlight_stays_inside_the_snippet():
    file = code_file(REFUNDS_PY)

    assert snippet_from_file(file, 4, 8, highlight=(6, 6)).highlight == (6, 6)
    assert snippet_from_file(file, 4, 8, highlight=(7, 50)).highlight == (7, 8)
    assert snippet_from_file(file, 4, 8, highlight=(1, 2)).highlight is None
    assert snippet_from_file(file, 4, 8, highlight=(7, 5)).highlight is None


@pytest.mark.parametrize(
    "path, language",
    [
        ("a/b.py", "python"),
        ("web/page.tsx", "tsx"),
        ("web/lib/money.ts", "typescript"),
        ("ios/Auth/Keychain.swift", "swift"),
        ("Makefile", "text"),
    ],
)
def test_language_comes_from_the_extension(path, language):
    assert language_for(path) == language


# search, read and slice together


async def test_code_evidence_copies_the_matching_lines(fake_github):
    (snippet,) = await code_evidence(reader(fake_github), "refund window days", max_files=2)

    assert snippet.path == REFUNDS
    assert snippet.code == lines(REFUNDS_PY, snippet.start_line, snippet.end_line)
    assert "REFUND_WINDOW_DAYS = 30" in snippet.code
    line = REFUNDS_PY.splitlines().index("REFUND_WINDOW_DAYS = 30") + 1
    assert snippet.highlight == (line, line)
    assert snippet.github_url.startswith(f"https://github.com/dropsubs/app/blob/{MAIN_SHA}/")


async def test_code_evidence_reads_at_most_max_files(fake_github):
    found = await code_evidence(reader(fake_github), "billing", max_files=1)

    assert len(found) == 1
    assert sum(1 for tool, _ in fake_github.calls if tool == "get_file_contents") == 1


async def test_code_evidence_reads_each_hit(fake_github):
    found = await code_evidence(reader(fake_github), "billing", max_files=2)

    assert {s.path for s in found} == {REFUNDS, "api/billing/fees.py"}
    fees = next(s for s in found if s.path == "api/billing/fees.py")
    assert fees.code == lines(FEES_PY, fees.start_line, fees.end_line)


def test_lines_are_numbered_as_github_numbers_them():
    # GitHub breaks lines only at newlines; a form feed or a Windows line ending is not a line.
    text = "first = 1\r\n\x0c\n# page two\nlast = 2\n"
    snippet = snippet_from_file(code_file(text), 3, 4)

    assert code_file(text).lines() == ["first = 1", "\x0c", "# page two", "last = 2"]
    assert snippet.code == "# page two\nlast = 2"
    assert snippet.github_url.endswith("#L3-L4")
