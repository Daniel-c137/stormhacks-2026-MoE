"""Keyterms for the worker's Scribe Realtime streams: the words transcription is biased toward,
so the agent's name (and with it the wake phrase), the team's names and its work come out
right. Fixed for a connection, so the worker fetches them when it joins and on every reconnect.
"""

import asyncio
import logging
from collections.abc import Iterable

from contracts import Meeting, TeamSettings, get_identity

from .agent.team_tools import jira_reader
from .config import Settings
from .store import Store

logger = logging.getLogger(__name__)

# Scribe Realtime's limits (ElevenLabs speech-to-text docs, Oct 2026).
MAX_KEYTERMS = 50
MAX_KEYTERM_CHARS = 20

# Dropped from the end of a term cut at a word boundary, so a cut "Refunds," is "Refunds"
TRAILING = ",;:-/&("


def fit(term: str) -> str | None:
    """The term trimmed, whitespace collapsed, and if still too long, cut after the last whole
    word that fits. None when blank or when not even the first word fits."""
    words = term.split()
    kept: list[str] = []
    for word in words:
        if len(" ".join([*kept, word])) > MAX_KEYTERM_CHARS:
            break
        kept.append(word)
    if len(kept) < len(words):
        kept = " ".join(kept).rstrip(TRAILING).split()
    return " ".join(kept) or None


def select(candidates: Iterable[str]) -> list[str]:
    """The candidates that fit, in order, the first of each ignoring case, at most
    MAX_KEYTERMS."""
    terms: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        term = fit(candidate)
        if term is None or term.casefold() in seen:
            continue
        seen.add(term.casefold())
        terms.append(term)
        if len(terms) == MAX_KEYTERMS:
            break
    return terms


async def meeting_keyterms(store: Store, settings: Settings, meeting: Meeting) -> list[str]:
    """In order: the agent's name and the team's own wake phrase (whole, then word by word);
    each member's full and first name; the repository's name and the Jira project key; the
    agenda's titles; then the keys of the project's unfinished Jira issues while there is room.
    A team without its own repository or project uses the deployment's, as the agent does."""
    team = await store.settings(meeting.team_id)
    candidates = [get_identity().agent_name]
    if team.wake_phrase:
        candidates += [team.wake_phrase, *team.wake_phrase.split()]
    for person in await store.members(meeting.team_id):
        candidates += [person.name, *person.name.split()[:1]]
    if repo := team.github.repo or settings.github_repo:
        candidates.append(repo.rstrip("/").rsplit("/", 1)[-1])
    if project := team.jira.project or settings.jira_project_key:
        candidates.append(project)
    if agenda := await store.agenda(meeting.id):
        candidates += [item.title for item in agenda.items]
    terms = select(candidates)
    if room := MAX_KEYTERMS - len(terms):
        terms = select([*terms, *await open_issue_keys(settings, team, room)])
    return terms


async def open_issue_keys(settings: Settings, team: TeamSettings, limit: int) -> list[str]:
    """Keys of the team's unfinished Jira issues; none when Jira is not configured. Keyterms are
    a nicety, so a failing or slow Jira is logged and skipped, never an error."""
    reader = jira_reader(settings, team)
    if isinstance(reader, str):
        return []
    try:
        issues = await asyncio.wait_for(reader.unfinished(limit), settings.connector_timeout)
    except Exception as e:  # captions must never wait on Jira
        logger.warning("Keyterms skip Jira issue keys: %s", str(e) or type(e).__name__)
        return []
    return [issue.key for issue in issues]
