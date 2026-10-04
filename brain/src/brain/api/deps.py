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
from ..agent.code import code_evidence
from ..agent.factcheck import FactChecker
from ..agent.pipeline import PipelineRunner, PostMeetingPipeline, ReportPipeline
from ..auth import NOT_CONFIGURED, AuthNotConfigured, InvalidToken, LoginLimiter, TokenVerifier
from ..config import Settings
from ..jira import JiraPusher, jira_config
from ..livekit_rooms import Rooms, rooms_from_settings
from ..llm import LLM, LLMError, LLMUnavailable, make_embedder, make_llm
from ..memory import MeetingMemory, PgMemoryStore, UnusableMemory
from ..speech import MeetingLocks
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
    """Gemini, then OpenRouter when Gemini is out of capacity (make_llm). Neither configured is
    a clear 503, never a silent mock; tests override this."""
    try:
        return make_llm(settings)
    except LLMUnavailable as e:
        raise HTTPException(status_code=503, detail=str(e)) from None


def get_llm_factory(settings: Settings = Depends(get_settings)) -> Callable[[], LLM]:
    """Makes the LLM when the write-up needs it, so ending a meeting never fails on an
    unconfigured model; the write-up step shows LLMUnavailable instead. Tests override it."""
    return lambda: make_llm(settings)


def get_memory(
    request: Request, settings: Settings = Depends(get_settings)
) -> MeetingMemory | UnusableMemory | None:
    """Meeting memory in Postgres (the store's pool on app.state.db_pool) with Gemini embeddings;
    None when either is missing; UnusableMemory when both are set but cannot work together, so
    the write-up reports the misconfiguration. Tests override it with an in-memory store."""
    pool = getattr(request.app.state, "db_pool", None)
    if pool is None:
        return None
    try:
        return MeetingMemory(make_embedder(settings), PgMemoryStore(pool))
    except LLMUnavailable:
        return None
    except ValueError as e:  # embedding dimensions the table cannot hold
        logger.warning("meeting memory is misconfigured: %s", e)
        return UnusableMemory(str(e))


def get_orchestrator(
    llm: LLM = Depends(get_llm),
    store: Store = Depends(get_store),
    memory: MeetingMemory | UnusableMemory | None = Depends(get_memory),
    settings: Settings = Depends(get_settings),
) -> ToolOrchestrator:
    """The agent for deliberate questions. Read-only; GitHub and Jira when configured."""
    return ToolOrchestrator(llm, store, settings=settings, memory=memory)


def get_fact_checker(
    store: Store = Depends(get_store),
    settings: Settings = Depends(get_settings),
    make_llm: Callable[[], LLM] = Depends(get_llm_factory),
    memory: MeetingMemory | UnusableMemory | None = Depends(get_memory),
) -> FactChecker:
    """Fact-checks for the worker's tick. The model is made only once a claim needs checking, so
    a quiet team never needs Gemini. Read-only; GitHub (code included) and Jira when configured.
    Tests override it."""
    return FactChecker(make_llm, store, settings=settings, memory=memory, code=code_evidence)


def get_pipeline(
    store: Store = Depends(get_store),
    settings: Settings = Depends(get_settings),
    make_llm: Callable[[], LLM] = Depends(get_llm_factory),
    memory: MeetingMemory | UnusableMemory | None = Depends(get_memory),
) -> PostMeetingPipeline:
    return ReportPipeline(store, make_llm, memory, settle_seconds=settings.pipeline_settle_seconds)


def get_runner(request: Request) -> PipelineRunner:
    """The app's background write-ups (created with the app)."""
    state = request.app.state
    if not hasattr(state, "pipeline_runner"):
        state.pipeline_runner = PipelineRunner()
    return state.pipeline_runner


def get_speech_locks(request: Request) -> MeetingLocks:
    """The app's per-meeting locks around making report audio."""
    state = request.app.state
    if not hasattr(state, "speech_locks"):
        state.speech_locks = MeetingLocks()
    return state.speech_locks


def get_rooms(settings: Settings = Depends(get_settings)) -> Rooms:
    """LiveKit's room service, to close a meeting's room when it ends."""
    return rooms_from_settings(settings)


def get_verifier(settings: Settings = Depends(get_settings)) -> TokenVerifier:
    return TokenVerifier.from_settings(settings)


def get_login_limiter(request: Request) -> LoginLimiter:
    """The app's count of failed logins per email (in process; reset on restart)."""
    state = request.app.state
    if not hasattr(state, "login_limiter"):
        state.login_limiter = LoginLimiter()
    return state.login_limiter


def _unauthorized(detail: str, error: str | None = None) -> HTTPException:
    challenge = f'Bearer error="{error}"' if error else "Bearer"
    return HTTPException(status_code=401, detail=detail, headers={"WWW-Authenticate": challenge})


async def current_user(
    authorization: str | None = Header(default=None),
    verifier: TokenVerifier = Depends(get_verifier),
    store: Store = Depends(get_store),
) -> Person:
    """Resolve the session token from POST /auth/login; every query is scoped to this user's
    team."""
    if not verifier.configured:
        raise HTTPException(status_code=503, detail=NOT_CONFIGURED)
    scheme, _, token = (authorization or "").partition(" ")
    token = token.strip()
    if scheme.lower() != "bearer" or not token or " " in token:
        raise _unauthorized("Missing bearer token")
    try:
        claims = verifier.verify(token)
    except AuthNotConfigured:
        raise HTTPException(status_code=503, detail=NOT_CONFIGURED) from None
    except InvalidToken:
        raise _unauthorized("Invalid or expired session", "invalid_token") from None
    try:
        return await store.person(claims["sub"])
    except NotFound:
        raise HTTPException(status_code=403, detail="not a workspace member") from None


ADMIN_ONLY = "Only an admin can do this"


async def require_admin(user: Person = Depends(current_user)) -> Person:
    """The caller, if they are an admin. current_user reads the person from the store on every
    request, so granting or revoking admin takes effect at once, whatever the session says."""
    if not user.is_admin:
        raise HTTPException(status_code=403, detail=ADMIN_ONLY)
    return user


HOST_OR_ADMIN = "Only the host or an admin can do this"


def is_host_or_admin(meeting: Meeting, person: Person) -> bool:
    """The one rule for what a meeting's host does: its host, or an admin. Callers must already
    know the person is on the meeting's team."""
    return meeting.host_id == person.id or person.is_admin


def host_or_admin(meeting: Meeting, user: Person) -> None:
    """403 unless the user hosts the meeting or is an admin, so a meeting whose host left can
    still be ended, have its invitees changed, its write-up retried and its tasks pushed."""
    if not is_host_or_admin(meeting, user):
        raise HTTPException(status_code=403, detail=HOST_OR_ADMIN)


def get_jira_pusher(settings: Settings = Depends(get_settings)) -> Callable[[], JiraPusher]:
    """Makes the pusher for an approved push; JiraUnavailable when Jira is not configured.
    A factory, so the route checks the team and the meeting before Jira's configuration.
    Tests override it with an in-process Jira."""
    return lambda: JiraPusher(jira_config(settings))


MIN_INTERNAL_TOKEN = 32


async def require_internal(
    x_internal_token: str | None = Header(default=None),
    settings: Settings = Depends(get_settings),
) -> None:
    """Guard for realtime -> brain calls (BRAIN_INTERNAL_TOKEN). A short token is treated as
    unset: /internal shares the board's port, so the token is the only thing protecting it."""
    expected = settings.brain_internal_token
    if not expected or len(expected) < MIN_INTERNAL_TOKEN:
        raise HTTPException(
            status_code=503,
            detail=f"BRAIN_INTERNAL_TOKEN must be set to at least {MIN_INTERNAL_TOKEN} characters",
        )
    # Bytes: compare_digest raises on non-ASCII str, which would turn a bad header into a 500.
    if not x_internal_token or not secrets.compare_digest(
        x_internal_token.encode(), expected.encode()
    ):
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
