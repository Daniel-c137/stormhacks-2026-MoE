"""Code snippets as evidence. A snippet is only ever a slice of a file GitHub returned, by line
numbers: a model may choose the lines, never the code. Answers (ask.py) and fact-checks use it."""

from pathlib import PurePosixPath
from uuid import uuid4

import anyio

from brain.github import CodeFile, GitHubError, GitHubReader
from brain.integrations import search_words
from contracts import CodeSnippet

MAX_SNIPPET_LINES = 40
CODE_FILES = 2
LINES_BEFORE = 6  # around the best matching line
LINES_AFTER = 12
LINES_WITHOUT_MATCH = 20  # from the top of a file whose path matched but no line did

LANGUAGES = {
    ".py": "python",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".js": "javascript",
    ".jsx": "jsx",
    ".mjs": "javascript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".kt": "kotlin",
    ".swift": "swift",
    ".rb": "ruby",
    ".php": "php",
    ".cs": "csharp",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".sql": "sql",
    ".sh": "bash",
    ".html": "html",
    ".css": "css",
    ".md": "markdown",
    ".json": "json",
    ".jsonl": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
}


def language_for(path: str) -> str:
    return LANGUAGES.get(PurePosixPath(path).suffix.lower(), "text")


def within(span: tuple[int, int] | None, start: int, end: int) -> tuple[int, int] | None:
    """The part of `span` inside start..end, or None when there is none."""
    if span is None or span[0] > span[1]:
        return None
    low, high = max(span[0], start), min(span[1], end)
    return (low, high) if low <= high else None


def snippet_from_file(
    file: CodeFile,
    start_line: int,
    end_line: int,
    *,
    highlight: tuple[int, int] | None = None,
    caption: str | None = None,
    max_lines: int = MAX_SNIPPET_LINES,
) -> CodeSnippet:
    """Lines start_line..end_line (1-based, inclusive) of the file, copied. A range reaching
    past the file is clamped to it and capped at max_lines; one entirely outside it, or
    backwards, raises ValueError. The highlight is kept to the part inside the snippet."""
    lines = file.lines()
    if start_line > end_line or end_line < 1 or start_line > len(lines):
        raise ValueError(
            f"lines {start_line}-{end_line} are not in {file.path} ({len(lines)} lines)"
        )
    start = max(start_line, 1)
    end = min(end_line, len(lines), start + max_lines - 1)
    at = f" at {file.ref[:7]}" if file.pinned else f" on {file.ref}"
    return CodeSnippet(
        id=str(uuid4()),
        path=file.path,
        start_line=start,
        end_line=end,
        language=language_for(file.path),
        code="\n".join(lines[start - 1 : end]),
        github_url=file.url(start, end),
        caption=caption or f"{file.path}, lines {start}-{end}{at}",
        highlight=within(highlight, start, end),
    )


def best_line(lines: list[str], query: str) -> int | None:
    """The first line with the most distinct query words in it, or None when none has any."""
    words = {w.lower() for w in search_words(query).split() if len(w) > 1}
    best, most = None, 0
    for number, line in enumerate(lines, start=1):
        low = line.lower()
        found = sum(1 for w in words if w in low)
        if found > most:
            best, most = number, found
    return best


def snippet_around(file: CodeFile, query: str) -> CodeSnippet:
    """The lines around the file's best match for the query, the match highlighted."""
    line = best_line(file.lines(), query)
    if line is None:
        return snippet_from_file(file, 1, LINES_WITHOUT_MATCH)
    return snippet_from_file(
        file, max(1, line - LINES_BEFORE), line + LINES_AFTER, highlight=(line, line)
    )


async def code_evidence(
    reader: GitHubReader, query: str, *, max_files: int = CODE_FILES
) -> list[CodeSnippet]:
    """Searches the repository's code and copies a snippet from each of the top max_files
    files, read at the reader's ref. Raises GitHubError when the search fails, or when every
    read does."""
    hits = await reader.search_code(query, limit=max_files)
    files: list[CodeFile | GitHubError | None] = [None] * len(hits)

    async def read(i: int, path: str) -> None:
        try:
            files[i] = await reader.read_file(path)
        except GitHubError as e:
            files[i] = e

    async with anyio.create_task_group() as group:
        for i, hit in enumerate(hits):
            group.start_soon(read, i, hit.path)

    snippets = []
    for file in files:
        if isinstance(file, CodeFile):
            try:
                snippets.append(snippet_around(file, query))
            except ValueError:
                continue  # an empty file
    errors = [f for f in files if isinstance(f, GitHubError)]
    if errors and not snippets:
        raise errors[0]
    return snippets
