import logging
import secrets
from collections.abc import Callable
from functools import cache
from typing import NoReturn

import httpx
from fastapi import Depends, Header, HTTPException, Request

from contracts import Answer, Meeting, Person, Team

from ..agent.ask import (
    MAX_HISTORY_TURNS,
    MAX_QUESTION_CHARS,
    MAX_TURN_CHARS,
    Question,
    ToolOrchestrator,
)
from ..auth import AuthNotConfigured, InvalidToken, KeysUnavailable, TokenVerifier
from ..config import Settings
from ..jira import JiraPusher, jira_config
from ..llm import LLM, LLMError, LLMUnavailable, make_embedder, make_llm
from ..memory import MeetingMemory, PgMemoryStore
from ..store import NotFound, Store

logger = logging.getLogger(__name__)


def not_implemented() -> NoReturn:
    raise HTTPException(status_code=501, detail="Not implemented")


@cache
def get_settings() -> Settings:
    return Settings()


def get_http_transport() -> httpx.AsyncBaseTransport | None:
    """The transport for outbound HTTP calls; None is the real network. Tests override it."""
    return None


async def get_store(request: Request) -> Store:
    """The Postgres store the app opened at startup (DATABASE_URL); 503 when there is none.
    Tests override this dependency."""
    store: Store | None = getattr(request.app.state, "store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="DATABASE_URL is not configured")
    return store


async def get_llm(settings: Settings = Depends(get_settings)) -> LLM:
    """Gemini. Unconfigured is a clear 503, never a silent mock; tests override this."""
    try:
        return make_llm(settings)
    except LLMUnavailable as e:
        raise HTTPException(status_code=503, detail=str(e)) from None


def get_memory(
    request: Request, settings: Settings = Depends(get_settings)
) -> MeetingMemory | None:
    """Meeting memory in Postgres (the store's pool on app.state.db_pool) with Gemini embeddings,
    or None when either is missing. Tests override it with an in-memory store."""
    pool = getattr(request.app.state, "db_pool", None)
    if pool is None:
        return None
    try:
        return MeetingMemory(make_embedder(settings), PgMemoryStore(pool))
    except LLMUnavailable:
        return None
    except ValueError as e:  # embedding dimensions the table cannot hold
        logger.warning("meeting memory is unavailable: %s", e)
        return None


def get_orchestrator(
    llm: LLM = Depends(get_llm),
    store: Store = Depends(get_store),
    memory: MeetingMemory | None = Depends(get_memory),
    settings: Settings = Depends(get_settings),
) -> ToolOrchestrator:
    """The agent for deliberate questions. Read-only; GitHub and Jira when configured."""
    return ToolOrchestrator(llm, store, settings=settings, memory=memory)


def get_verifier(request: Request, settings: Settings = Depends(get_settings)) -> TokenVerifier:
    """One verifier per app and auth config, so the JWKS cache outlives a request."""
    state = request.app.state
    if not hasattr(state, "token_verifiers"):
        state.token_verifiers = {}
    cache: dict[tuple, TokenVerifier] = state.token_verifiers
    key = (settings.supabase_url, settings.supabase_jwt_secret, settings.supabase_jwt_audience)
    if key not in cache:
        cache[key] = TokenVerifier.from_settings(settings)
    return cache[key]


def _unauthorized(detail: str, error: str | None = None) -> HTTPException:
    challenge = f'Bearer error="{error}"' if error else "Bearer"
    return HTTPException(status_code=401, detail=detail, headers={"WWW-Authenticate": challenge})


NOT_CONFIGURED = "Supabase auth is not configured"


async def current_user(
    authorization: str | None = Header(default=None),
    verifier: TokenVerifier = Depends(get_verifier),
    store: Store = Depends(get_store),
) -> Person:
    """Resolve the Supabase session; every query is scoped to this user's team."""
    if not verifier.configured:
        raise HTTPException(status_code=503, detail=NOT_CONFIGURED)
    scheme, _, token = (authorization or "").partition(" ")
    token = token.strip()
    if scheme.lower() != "bearer" or not token or " " in token:
        raise _unauthorized("Missing bearer token")
    try:
        claims = await verifier.verify(token)
    except AuthNotConfigured:
        raise HTTPException(status_code=503, detail=NOT_CONFIGURED) from None
    except KeysUnavailable:
        raise HTTPException(
            status_code=503, detail="Supabase signing keys are unavailable"
        ) from None
    except InvalidToken:
        raise _unauthorized("Invalid or expired session", "invalid_token") from None
    try:
        return await store.person(claims["sub"])
    except NotFound:
        raise HTTPException(status_code=403, detail="not a workspace member") from None


def get_jira_pusher(settings: Settings = Depends(get_settings)) -> Callable[[], JiraPusher]:
    """Makes the pusher for an approved push; JiraUnavailable when Jira is not configured.
    A factory, so the route checks the team and the meeting before Jira's configuration.
    Tests override it with an in-process Jira."""
    return lambda: JiraPusher(jira_config(settings))


async def require_internal(
    x_internal_token: str | None = Header(default=None),
    settings: Settings = Depends(get_settings),
) -> None:
    """Guard for realtime -> brain calls (BRAIN_INTERNAL_TOKEN)."""
    expected = settings.brain_internal_token
    if not expected:
        raise HTTPException(status_code=503, detail="BRAIN_INTERNAL_TOKEN is not configured")
    if not x_internal_token or not secrets.compare_digest(x_internal_token, expected):
        raise HTTPException(status_code=401, detail="Invalid internal token")


async def user_team(store: Store, user: Person) -> Team:
    try:
        return await store.team_for_user(user.id)
    except NotFound:
        raise HTTPException(status_code=403, detail="You are not on a team") from None


async def team_meeting(store: Store, user: Person, meeting_id: str) -> Meeting:
    """The meeting if it belongs to the user's team. Other teams' meetings are a plain 404."""
    team = await user_team(store, user)
    try:
        meeting = await store.meeting(meeting_id)
    except NotFound:
        meeting = None
    if meeting is None or meeting.team_id != team.id:
        raise HTTPException(status_code=404, detail="Meeting not found")
    return meeting


async def ask_agent(orchestrator: ToolOrchestrator, question: Question) -> Answer:
    """The agent's answer; 422 for an oversized question or history, 502 when the model fails.
    Nothing about the question is kept."""
    if len(question.text) > MAX_QUESTION_CHARS:
        raise HTTPException(
            status_code=422, detail=f"Questions are at most {MAX_QUESTION_CHARS} characters"
        )
    if len(question.history) > MAX_HISTORY_TURNS:
        raise HTTPException(status_code=422, detail=f"History is at most {MAX_HISTORY_TURNS} turns")
    if any(len(turn.text) > MAX_TURN_CHARS for turn in question.history):
        raise HTTPException(
            status_code=422, detail=f"History turns are at most {MAX_TURN_CHARS} characters"
        )
    try:
        return await orchestrator.ask(question)
    except LLMError as e:
        raise HTTPException(status_code=502, detail=f"Could not answer: {e}") from e
