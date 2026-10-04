"""Fixtures for the live end-to-end meeting suite (test_live_meeting.py): a seeded embedded
Postgres, the GitHub and Jira MCP servers the brain reads, the brain itself as a real uvicorn
subprocess with its own sign-in (AUTH_SECRET), and a login with a password for each of the
cast."""

import json
import os
import secrets
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import anyio
import httpx
import pytest
from conftest import JIRA_ACCOUNTS, FakeGitHub, FakeJira, serve_mcp
from fact_check_support import merged_after_release
from pg_support import fresh_database

from brain.auth import hash_password
from brain.config import Settings
from brain.db import migrate
from brain.jira import JiraConfig, JiraIssue, JiraReader
from brain.pg_store import PostgresStore
from contracts import GitHubSettings, JiraSettings, Person, Team, TeamSettings

try:  # the demo world's mock Jira over mock-data (#71)
    from world.jira_mcp import load_world
    from world.jira_mcp import server as world_jira
except ImportError:
    world_jira = None

REPO_ROOT = Path(__file__).resolve().parents[3]
STANDUP = json.loads((REPO_ROOT / "brain/tests/fixtures/standup.json").read_text())
TEAM_TZ = "America/Vancouver"
JIRA_SITE = "https://dropsubs.atlassian.net"
JIRA_PROJECT = "DS"
# DS-104 as the tests' FakeJira serves it; the world's mock Jira has its own DS-104.
FAKE_JIRA_ISSUES = [
    {
        "key": "DS-104",
        "fields": {
            "summary": "Fix the waitlist email exploit",
            "status": {"name": "In Progress"},
            "assignee": {"displayName": "Carol Jensen"},
            "priority": {"name": "High"},
            "description": "Signup emails could be sent to arbitrary addresses.",
        },
    }
]
# Names the demo world's mock Jira knows, so a pushed task can be assigned to its owner.
WORLD_NAMES = {"p-alice": "Danial", "p-bob": "Mohammad Reza", "p-carol": "Reyhaneh"}


@dataclass(frozen=True)
class Cast:
    """The standup's people: alice hosts, bob claims PR 41 is released and owns the refunds,
    carol asks about DS-104 in private."""

    alice: Person
    bob: Person
    carol: Person

    @property
    def everyone(self) -> tuple[Person, Person, Person]:
        return (self.alice, self.bob, self.carol)

    def name(self, person_id: str | None) -> str | None:
        return next((p.name for p in self.everyone if p.id == person_id), None)


def standup_cast(renames: dict[str, str] | None = None) -> Cast:
    """The standup fixture's members, optionally renamed by id."""
    people = []
    for member in STANDUP["members"]:
        person = Person(**member, email=f"{member['short'].lower()}@dropsubs.dev")
        if renames and person.id in renames:
            name = renames[person.id]
            initials = "".join(word[0] for word in name.split())
            person = person.model_copy(
                update={"name": name, "short": name, "initials": initials, "email": None}
            )
        people.append(person)
    return Cast(*people)


@dataclass
class JiraSource:
    """The Jira MCP server the brain reads and pushes to. `kind` is "world" (the demo world's
    mock Jira over mock-data) or "fake" (the tests' FakeJira), and `accounts` the display names
    it can assign issues to."""

    url: str
    kind: str
    accounts: set[str]
    fake: FakeJira | None = None

    def read_back(self, key: str) -> JiraIssue | None:
        """An issue as the server now holds it, bypassing the brain."""
        if self.fake is not None:
            created = next((c for c in self.fake.created if c["key"] == key), None)
            if created is None:
                return None
            names = {a["accountId"]: a["displayName"] for a in self.fake.accounts}
            return JiraIssue(
                key=key,
                summary=created["summary"],
                assignee=names.get(created["assignee_account_id"]),
            )
        config = JiraConfig(
            mcp_url=self.url, cloud_id=JIRA_SITE, project_key=JIRA_PROJECT, base_url=JIRA_SITE
        )
        return anyio.run(JiraReader(config).get, key)


@dataclass
class GitHubSource:
    """The GitHub MCP server the brain reads and the repository it holds. `kind` is "fake": the
    tests' FakeGitHub, whose code, PR #41 and releases the flow's GitHub steps expect."""

    url: str
    repo: str
    kind: str


@dataclass(frozen=True)
class Login:
    email: str
    password: str


@dataclass
class Brain:
    url: str
    internal_token: str
    log: Path


@pytest.fixture(scope="module")
def jira_source(tmp_path_factory) -> Iterator[JiraSource]:
    """The demo world's mock Jira when the world package is installed, else the FakeJira."""
    if world_jira is not None:
        with pytest.MonkeyPatch.context() as env:
            env.setenv("WORLD_OVERLAY_DIR", str(tmp_path_factory.mktemp("world-overlay")))
            env.setenv("WORLD_SNAPSHOT", "demo")
            accounts = {a["displayName"] for a in load_world().accounts}
            with serve_mcp(world_jira) as url:
                yield JiraSource(url=url, kind="world", accounts=accounts)
        return
    jira = FakeJira(issue_reads=True)
    jira.issues = [dict(issue) for issue in FAKE_JIRA_ISSUES]
    with serve_mcp(jira.server) as url:
        yield JiraSource(
            url=url, kind="fake", accounts={a["displayName"] for a in JIRA_ACCOUNTS}, fake=jira
        )


@pytest.fixture(scope="module")
def github_source() -> Iterator[GitHubSource]:
    """The one place GitHub is chosen. The demo world's mock GitHub (#74) can replace the fake
    here once it lands; the fact-check and code steps then need its equivalents of PR #41 and
    the refund window."""
    github = FakeGitHub()
    merged_after_release(github)
    with serve_mcp(github.server) as url:
        yield GitHubSource(url=url, repo=github.full_name, kind="fake")


@pytest.fixture(scope="module")
def cast(jira_source) -> Cast:
    """The world's mock Jira knows its own people, so the standup speaks with their names."""
    return standup_cast(WORLD_NAMES if jira_source.kind == "world" else None)


@pytest.fixture(scope="module")
def team(github_source) -> Team:
    return Team(id="team-dropsubs", name="DropSubs", member_ids=[], github_repo=github_source.repo)


def login_email(person: Person) -> str:
    """The cast sign in with their email; people renamed for the world's Jira have none."""
    return person.email or f"{person.id}@dropsubs.dev"


@pytest.fixture(scope="module")
def logins(cast) -> dict[str, Login]:
    """Each of the cast's email and a password made for this run, by person id."""
    return {p.id: Login(login_email(p), secrets.token_urlsafe(16)) for p in cast.everyone}


@pytest.fixture(scope="module")
def live_dsn(pg_server, team, cast, logins, github_source) -> Iterator[str]:
    """A fresh database with every migration applied and the team, its people (each with a
    login) and its settings (America/Vancouver) seeded."""

    async def seed(dsn: str) -> None:
        await migrate(dsn)
        store = PostgresStore(dsn)
        await store.create_team(team)
        for person in cast.everyone:
            await store.upsert_person(person, team.id)
            login = logins[person.id]
            await store.set_login(person.id, login.email, hash_password(login.password))
        await store.save_settings(
            TeamSettings(
                team_id=team.id,
                github=GitHubSettings(repo=github_source.repo),
                jira=JiraSettings(site=JIRA_SITE, project=JIRA_PROJECT),
                sensitivity="balanced",
                interrupt_minutes=5,
                timezone=TEAM_TZ,
            )
        )

    with fresh_database(pg_server) as dsn:
        anyio.run(seed, dsn)
        yield dsn


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def listen_enabled(settings: Settings) -> bool:
    """Speech spends ElevenLabs credits: only with E2E_LISTEN=1 and speech configured."""
    configured = settings.elevenlabs_api_key and settings.elevenlabs_voice_id
    return os.environ.get("E2E_LISTEN") == "1" and bool(
        configured and settings.elevenlabs_tts_model
    )


@pytest.fixture(scope="module")
def brain(live_dsn, jira_source, github_source, tmp_path_factory) -> Iterator[Brain]:
    """The brain as a uvicorn subprocess on the seeded database. Gemini, embedding, ElevenLabs
    (and any OpenRouter) settings come from the environment and .env as usual; everything else
    points at this run's local servers. LiveKit is deliberately unreachable. Without E2E_LISTEN
    the speech model is blanked, so nothing can spend ElevenLabs credits."""
    auth_secret, internal_token = secrets.token_urlsafe(48), secrets.token_urlsafe(32)
    env = {
        **os.environ,
        "DATABASE_URL": live_dsn,
        "AUTH_SECRET": auth_secret,
        "BRAIN_INTERNAL_TOKEN": internal_token,
        "GITHUB_MCP_URL": github_source.url,
        "GITHUB_REPO": github_source.repo,
        "GITHUB_TOKEN": "",
        "JIRA_MCP_URL": jira_source.url,
        "JIRA_PROJECT_KEY": JIRA_PROJECT,
        "JIRA_BASE_URL": JIRA_SITE,
        "JIRA_CLOUD_ID": "",
        "LIVEKIT_URL": f"ws://127.0.0.1:{free_port()}",
        "LIVEKIT_API_KEY": "e2e-key",
        "LIVEKIT_API_SECRET": secrets.token_urlsafe(32),
        "PIPELINE_SETTLE_SECONDS": "2",
    }
    if not listen_enabled(Settings()):
        env["ELEVENLABS_TTS_MODEL"] = ""
    port = free_port()
    log = tmp_path_factory.mktemp("brain") / "brain.log"
    with log.open("w") as out:
        proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "brain.main:app", "--port", str(port)],
            cwd=REPO_ROOT,
            env=env,
            stdout=out,
            stderr=subprocess.STDOUT,
        )
    url = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 60
        while True:
            if proc.poll() is not None:
                pytest.fail(f"the brain exited with {proc.returncode}; see {log}")
            try:
                httpx.get(f"{url}/docs", timeout=1)
                break
            except httpx.HTTPError:
                if time.monotonic() > deadline:
                    pytest.fail(f"the brain did not start within 60 s; see {log}")
                time.sleep(0.3)
        print(
            f"\nbrain on {url} (log: {log}); Jira: {jira_source.kind}; GitHub: {github_source.kind}"
        )
        yield Brain(url=url, internal_token=internal_token, log=log)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()


def log_in(client: httpx.Client, login: Login) -> httpx.Response:
    """POST /auth/login, as the board signs in."""
    return client.post("/auth/login", json={"email": login.email, "password": login.password})


def session(response: httpx.Response) -> dict[str, str]:
    """Authorization headers for the session a successful login returned."""
    return {"Authorization": f"Bearer {response.json()['token']}"}


def elevenlabs_characters(settings: Settings) -> int | None:
    """Characters the ElevenLabs account has used this period (GET /v1/user/subscription, which
    costs nothing), or None when it cannot be read."""
    if not settings.elevenlabs_api_key:
        return None
    try:
        response = httpx.get(
            f"{settings.elevenlabs_api_url}/v1/user/subscription",
            headers={"xi-api-key": settings.elevenlabs_api_key},
            timeout=15,
        )
    except httpx.HTTPError:
        return None
    if response.status_code != 200:
        return None
    return int(response.json().get("character_count", 0))
