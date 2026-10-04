"""Live: one meeting end to end through the real brain, against real Gemini.

The brain runs as a uvicorn subprocess on embedded Postgres 16 + pgvector (pgserver, every
migration applied), with Supabase sessions signed locally (HS256) and an internal token for the
worker's calls. Jira is the demo world's mock Jira over mock-data (the tests' FakeJira if the
world package is missing), with the standup's people renamed to the people it knows. GitHub is
the tests' FakeGitHub, where PR #41 merged after the latest release. LiveKit is unreachable on
purpose. The team is on America/Vancouver. The steps run in order and share one meeting: schedule
and agenda, join, transcript ingest, timekeeping and fact-check ticks, questions in the meeting,
end, the write-up, review and push to Jira, then history and settings. They assert invariants,
not wording. A step whose earlier step failed is skipped, naming what it needed.

Run from the repo root (deselected by default; skipped without Gemini and embedding settings):

    GEMINI_MODEL=gemini-3.5-flash-lite GEMINI_FALLBACK_MODELS=gemini-3.8-flash,gemini-3.7-flash \\
    GEMINI_EMBEDDING_MODEL=gemini-embedding-001 GEMINI_EMBEDDING_DIM=768 \\
    uv run pytest brain/tests/e2e/test_live_meeting.py -m live -v -s

It takes about 2-3 minutes, plus a minute for each write-up retry. Cost: roughly 15-20 Gemini
generate calls and a handful of embedding calls, a large share of the free tier's per-minute
quota, so run it once and space runs at least a minute apart. If the write-up hits the quota,
the host retries it through POST /meetings/{id}/report/retry after a minute's wait, up to three
times. An OpenRouter fallback (#61) is landing separately; once it does, the brain picks it up
from any OPENROUTER_* settings in the environment or .env, with no change here.

ElevenLabs: listing voices is free and runs whenever ELEVENLABS_API_KEY is set. Listen (the
summary read aloud) spends credits, so it runs only with E2E_LISTEN=1 and ELEVENLABS_API_KEY,
ELEVENLABS_VOICE_ID and ELEVENLABS_TTS_MODEL set. It makes one speech request of at most
TTS_CHAR_BUDGET characters (one summary, a few hundred characters; an earlier run's 288
characters cost 63 credits) and reads the account's character usage before and after,
which is free. Without E2E_LISTEN the brain runs with no speech model, so it cannot spend any.
"""

import json
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import httpx
import pytest
from live_support import (  # noqa: F401  (shared fixtures)
    STANDUP,
    TEAM_TZ,
    Brain,
    Cast,
    JiraSource,
    brain,
    cast,
    elevenlabs_characters,
    github_source,
    jira_source,
    listen_enabled,
    live_dsn,
    session,
    team,
)

from brain.config import Settings
from contracts import get_identity

settings = Settings()

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not (settings.gemini_api_key and settings.gemini_model),
        reason="set GEMINI_API_KEY and GEMINI_MODEL to run against Gemini",
    ),
    pytest.mark.skipif(
        not (settings.gemini_embedding_model and settings.gemini_embedding_dim),
        reason="set GEMINI_EMBEDDING_MODEL and GEMINI_EMBEDDING_DIM for the write-up's memory",
    ),
]

TTS_CHAR_BUDGET = 1500  # the most characters one run may send to ElevenLabs text-to-speech
WRITE_UP_TIMEOUT = 240  # seconds to wait for one write-up attempt
WRITE_UP_RETRIES = 3
QUOTA_WAIT = 60  # seconds before a retry: the free tier's quota is per minute
BOB_CLAIM = "And the refund fix from PR 41 is already released in v0.9.3."
PRIVATE_QUESTION = "Is DS-104 done in Jira yet?"
HOME_QUESTION = "What did we decide about the waitlist email, and when?"


@dataclass
class Flow:
    """What the steps share: the client, everyone's headers, and what earlier steps made."""

    client: httpx.Client
    cast: Cast
    jira: JiraSource
    alice: dict[str, str]
    bob: dict[str, str]
    carol: dict[str, str]
    worker: dict[str, str]
    meeting: dict[str, Any] | None = None
    segments: list[dict[str, Any]] = field(default_factory=list)
    contradicted: list[dict[str, Any]] | None = None
    ended_at: float | None = None
    write_up_done: bool = False
    report: dict[str, Any] | None = None
    home_answer: dict[str, Any] | None = None

    @property
    def mid(self) -> str:
        return need(self.meeting, "the scheduled meeting")["id"]

    @property
    def code(self) -> str:
        return need(self.meeting, "the scheduled meeting")["code"]


def need(value, what: str):
    """An earlier step's result, or skip this step: its own failure is already reported."""
    if not value:
        pytest.skip(f"needs {what}, which an earlier step did not produce")
    return value


def show(step: str, detail: str) -> None:
    print(f"\n  [{step}] {detail}")


def short(response: httpx.Response, n: int = 200) -> str:
    return f"{response.status_code} {response.text[:n]}"


@pytest.fixture(scope="module")
def flow(brain: Brain, cast: Cast, jira_source: JiraSource):  # noqa: F811
    with httpx.Client(base_url=brain.url, timeout=120) as client:
        yield Flow(
            client=client,
            cast=cast,
            jira=jira_source,
            alice=session(brain, cast.alice),
            bob=session(brain, cast.bob),
            carol=session(brain, cast.carol),
            worker={"X-Internal-Token": brain.internal_token},
        )


def standup_segments(flow: Flow, mid: str) -> list[dict[str, Any]]:
    """The standup fixture in this meeting with the cast's names, then Bob's claim that PR 41
    is released, which the fake GitHub contradicts."""
    segments = [
        {
            **s,
            "meeting_id": mid,
            "seg_id": f"{mid}-{s['seg_id']}",
            "speaker_name": flow.cast.name(s["speaker_id"]) or s["speaker_name"],
        }
        for s in STANDUP["segments"]
    ]
    bob = flow.cast.bob
    return [
        *segments,
        {
            "seg_id": f"{mid}-seg-7",
            "meeting_id": mid,
            "speaker_id": bob.id,
            "speaker_name": bob.name,
            "is_final": True,
            "t_start": 40,
            "t_end": 47,
            "text": BOB_CLAIM,
        },
    ]


class TestLiveMeeting:
    """Before, during and after one meeting, in order."""

    def test_01_sessions(self, flow: Flow):
        r = flow.client.get("/me", headers=flow.alice)
        assert r.status_code == 200 and r.json()["id"] == flow.cast.alice.id, short(r)
        r = flow.client.get("/me", headers={"Authorization": "Bearer not-a-token"})
        assert r.status_code == 401, f"a bad token must be refused: {short(r)}"

    def test_02_schedule_with_an_agenda(self, flow: Flow):
        start = datetime.now(UTC) + timedelta(minutes=2)
        r = flow.client.post(
            "/meetings",
            headers=flow.alice,
            json={
                "title": STANDUP["title"],
                "scheduled_start": start.isoformat(),
                "duration_min": 15,
                "invitee_ids": [flow.cast.bob.id, flow.cast.carol.id],
            },
        )
        assert r.status_code in (200, 201) and r.json()["status"] == "scheduled", short(r)
        flow.meeting = r.json()
        show("schedule", f"code={flow.code}")

        items = ["Waitlist email", "Refunds for the double charge", "Launch date"]
        r = flow.client.put(
            f"/meetings/{flow.mid}/agenda",
            headers=flow.alice,
            json={"items": [{"title": t, "minutes": 5} for t in items]},
        )
        assert r.status_code == 200 and len(r.json()["items"]) == 3, f"lobby edit: {short(r)}"

    def test_03_agenda_rewrite_with_gemini(self, flow: Flow):
        r = flow.client.post(
            "/agenda/rewrite", headers=flow.bob, json={"text": "redis vs postgres??"}
        )
        assert r.status_code == 200 and r.json()["text"].strip(), short(r)
        show("rewrite", r.json()["text"])

    def test_04_agenda_suggestions_without_past_meetings(self, flow: Flow):
        r = flow.client.post(f"/meetings/{flow.mid}/agenda/suggest", headers=flow.alice)
        assert r.status_code == 200, short(r)
        show("suggest", f"{len(r.json().get('items', []))} items")

    def test_05_join(self, flow: Flow):
        url = f"/meetings/join/{flow.code}"
        r = flow.client.post(url, headers=flow.bob)
        assert r.status_code == 409, f"a guest before the host starts must wait: {short(r)}"
        r = flow.client.post(url, headers=flow.alice)
        joined = r.json() if r.status_code == 200 else {}
        assert joined.get("meeting", {}).get("status") == "live" and joined.get("token"), (
            f"the host starts the meeting with a LiveKit token: {short(r)}"
        )
        for who, headers in (("bob", flow.bob), ("carol", flow.carol)):
            r = flow.client.post(url, headers=headers)
            assert r.status_code == 200, f"{who} joins: {short(r)}"

    def test_06_worker_ingests_final_segments(self, flow: Flow):
        url = f"/internal/meetings/{flow.mid}/segments"
        segments = standup_segments(flow, flow.mid)
        r = flow.client.post(url, headers=flow.worker, json={"segments": segments})
        assert r.status_code == 204, f"ingest: {short(r)}"
        r = flow.client.post(url, headers=flow.worker, json={"segments": segments})
        assert r.status_code == 204, f"an identical resend: {short(r)}"
        r = flow.client.get(f"/meetings/{flow.mid}/transcript", headers=flow.alice)
        assert r.status_code == 200 and len(r.json()) == len(segments), (
            f"a resend is not stored twice: {len(r.json())} segments for {len(segments)}"
        )
        r = flow.client.post(
            url, headers={"X-Internal-Token": "x" * 40}, json={"segments": segments}
        )
        assert r.status_code == 401, f"a wrong internal token must be refused: {short(r)}"
        flow.segments = segments

    def test_07_timekeeping_tick_with_gemini(self, flow: Flow):
        need(flow.segments, "the ingested transcript")
        time.sleep(1)
        r = flow.client.post(
            f"/internal/meetings/{flow.mid}/agenda/track", headers=flow.worker, json={"now": 55}
        )
        assert r.status_code == 200, short(r, 300)
        body = r.json()
        agenda = body.get("agenda", {})
        items = agenda.get("items", [])
        current = next((i for i in items if i["id"] == agenda.get("current_item_id")), None)
        show(
            "timekeeping",
            f"current={current['title'] if current else None} "
            + ", ".join(f"{i['title']}={i['status']}/{i.get('discussed_s', 0):.0f}s" for i in items)
            + f" nudges={len(body.get('nudges', []))}",
        )
        covered = [i for i in items if i["status"] == "covered"]
        assert covered or current, "the discussion moved no agenda item"
        assert current is None or current.get("discussed_s", 0) > 0, "the current item has no time"
        assert agenda.get("tracked_until") is not None, "the tick did not record its progress"

    def test_08_fact_check_tick_with_gemini(self, flow: Flow):
        need(flow.segments, "the ingested transcript")
        r = flow.client.post(
            f"/internal/meetings/{flow.mid}/fact-check", headers=flow.worker, json={"now": 55}
        )
        assert r.status_code == 200, short(r, 300)
        body = r.json()
        checks = body.get("checks", [])
        show(
            "fact-check",
            "; ".join(
                f"{k['verdict']}/{k['severity']}/{k['confidence']:.2f}/{k['visibility']}"
                f"/hand={k['raised_hand']}: {k['claim'][:60]}"
                for k in checks
            ),
        )
        flow.contradicted = [k for k in checks if k["verdict"] == "contradicted"]
        found = [k for k in flow.contradicted if "41" in k["claim"]]
        assert found, "Bob's claim that PR 41 is released in v0.9.3 was not contradicted"
        check = found[0]
        labels = [s["label"] for s in check["sources"]]
        assert any(s["kind"].startswith("github") for s in check["sources"]), (
            f"the contradiction cites no GitHub source: {labels}"
        )
        if check["raised_hand"]:
            assert body.get("agent_state", {}).get("state") == "hand_raised", (
                f"the hand is raised and nothing is spoken: {body.get('agent_state')}"
            )

    def test_09_ask_in_the_meeting_public(self, flow: Flow):
        need(flow.segments, "the ingested transcript")
        r = flow.client.post(
            f"/meetings/{flow.mid}/ask",
            headers=flow.bob,
            json={
                "question": "What did we decide about the waitlist email?",
                "visibility": "public",
            },
        )
        assert r.status_code == 200, short(r)
        answer = r.json()
        show(
            "ask public",
            f"sources={[s['label'] for s in answer['sources']]} {answer['text'][:160]!r}",
        )
        assert any(s["kind"] == "meeting" for s in answer["sources"]), "no meeting source cited"

    def test_10_ask_in_the_meeting_private_from_jira(self, flow: Flow):
        r = flow.client.post(
            f"/meetings/{flow.mid}/ask",
            headers=flow.carol,
            json={"question": PRIVATE_QUESTION, "visibility": "private"},
        )
        assert r.status_code == 200, short(r)
        answer = r.json()
        show(
            "ask private",
            f"sources={[s['label'] for s in answer['sources']]} "
            f"unavailable={answer['unavailable']} {answer['text'][:160]!r}",
        )
        assert any(s["kind"] == "jira_issue" for s in answer["sources"]), (
            f"DS-104 was not read from Jira ({flow.jira.kind})"
        )

    def test_11_voice_invocation_answered_from_the_code(self, flow: Flow):
        segments = need(flow.segments, "the ingested transcript")
        alice = flow.cast.alice
        invocation = {
            "id": "inv-1",
            "meeting_id": flow.mid,
            "via": "voice",
            "visibility": "public",
            "asked_by_id": alice.id,
            "asked_by_name": alice.name,
            "t": 50,
            "question": f"Hey {get_identity().agent_name}, what's the refund window in the code?",
        }
        r = flow.client.post(
            f"/internal/meetings/{flow.mid}/invoke",
            headers=flow.worker,
            json={"invocation": invocation, "recent_segments": segments[-3:]},
        )
        assert r.status_code == 200, short(r)
        answer = r.json().get("answer", {})
        snippets = answer.get("snippets", [])
        show(
            "voice",
            f"snippet={snippets[0]['github_url'] if snippets else None}"
            f" {answer.get('text', '')[:140]!r}",
        )
        assert snippets and "REFUND_WINDOW_DAYS" in snippets[0]["code"], (
            "the answer quotes no snippet with the refund window"
        )

    def test_12_nothing_to_listen_to_while_live(self, flow: Flow):
        r = flow.client.get(f"/meetings/{flow.mid}/report/audio", headers=flow.alice)
        assert r.status_code in (404, 409), short(r)

    def test_13_voices_are_listed_for_free(self, flow: Flow):
        if not settings.elevenlabs_api_key:
            pytest.skip("set ELEVENLABS_API_KEY to list voices (free)")
        r = flow.client.get("/voices", headers=flow.alice)
        assert r.status_code == 200 and r.json(), short(r)
        listed = any(v["id"] == settings.elevenlabs_voice_id for v in r.json())
        show("voices", f"{len(r.json())} voices; the configured voice listed: {listed}")

    def test_14_end(self, flow: Flow):
        need(flow.segments, "the ingested transcript")
        r = flow.client.post(f"/meetings/{flow.mid}/end", headers=flow.bob)
        assert r.status_code == 403, f"only the host may end: {short(r)}"
        started = time.monotonic()
        r = flow.client.post(f"/meetings/{flow.mid}/end", headers=flow.alice)
        took = time.monotonic() - started
        assert r.status_code == 200 and r.json()["status"] == "processing", short(r)
        assert took < 10, f"ending waited {took:.1f}s: the write-up must run in the background"
        flow.ended_at = started
        carol = flow.cast.carol
        late = {
            "seg_id": f"{flow.mid}-late",
            "meeting_id": flow.mid,
            "speaker_id": carol.id,
            "speaker_name": carol.name,
            "is_final": True,
            "t_start": 60,
            "t_end": 63,
            "text": "Thanks all, bye.",
        }
        r = flow.client.post(
            f"/internal/meetings/{flow.mid}/segments",
            headers=flow.worker,
            json={"segments": [late]},
        )
        assert r.status_code == 204, f"a late final segment lands while processing: {short(r)}"

    def test_15_write_up_with_gemini_and_embeddings(self, flow: Flow):
        need(flow.ended_at, "the ended meeting")

        def wait() -> dict[str, Any]:
            progress: dict[str, Any] = {}
            deadline = time.monotonic() + WRITE_UP_TIMEOUT
            while time.monotonic() < deadline:
                r = flow.client.get(f"/meetings/{flow.mid}/report/progress", headers=flow.alice)
                progress = r.json() if r.status_code == 200 else {}
                if progress.get("done") or progress.get("error"):
                    break
                time.sleep(2)
            return progress

        progress = wait()
        for attempt in range(1, WRITE_UP_RETRIES + 1):
            if not progress.get("error"):
                break
            # The free tier's per-minute quota or an overloaded model: the meeting stays
            # processing with the error recorded, and the host retries, as the board offers.
            show("write-up", f"failed ({progress['error']}); host retry {attempt} in {QUOTA_WAIT}s")
            time.sleep(QUOTA_WAIT)
            r = flow.client.post(f"/meetings/{flow.mid}/report/retry", headers=flow.alice)
            assert r.status_code == 202, f"host retry {attempt}: {short(r)}"
            progress = wait()
        show(
            "write-up",
            f"steps={progress.get('steps')} after {time.monotonic() - flow.ended_at:.0f}s",
        )
        assert progress.get("done") and not progress.get("error"), (
            f"the write-up did not finish: error={progress.get('error')}"
        )
        meeting = flow.client.get(f"/meetings/{flow.mid}", headers=flow.alice).json()
        assert meeting["status"] == "needs_review", meeting["status"]
        flow.write_up_done = True

    def test_16_report_invariants(self, flow: Flow):
        need(flow.write_up_done, "the finished write-up")
        r = flow.client.get(f"/meetings/{flow.mid}/report", headers=flow.alice)
        assert r.status_code == 200, short(r)
        report = flow.report = r.json()
        transcript = flow.client.get(f"/meetings/{flow.mid}/transcript", headers=flow.alice)
        spoken = " ".join(s["text"] for s in transcript.json())
        decisions, tasks = report.get("decisions", []), report.get("tasks", [])
        show(
            "report",
            f"{len(decisions)} decisions, {len(tasks)} tasks "
            f"{[(t['title'][:40], t.get('owner_id')) for t in tasks]} "
            f"summary={report.get('summary', '')[:160]!r}",
        )

        quotes = [x["quote"] for x in [*decisions, *tasks] if x.get("quote")]
        invented = [q for q in quotes if q not in spoken]
        assert not invented, f"quotes that are not transcript text: {invented}"
        members = {p.id for p in flow.cast.everyone}
        owners = [t.get("owner_id") for t in tasks]
        assert all(o is None or o in members for o in owners), (
            f"every task owner is a participant, never the agent: {owners}"
        )
        if any("41" in k["claim"] for k in flow.contradicted or []):
            claims = [f["claim"] for f in report.get("fact_checks", [])]
            assert any("41" in c for c in claims), f"public fact-checks missing: {claims}"
        assert PRIVATE_QUESTION not in json.dumps(report), "the private question is in the report"
        assert PRIVATE_QUESTION not in spoken, "the private question is in the transcript"
        assert report.get("topics") or report.get("summary"), "no topics and no summary"

    def test_17_listen_reads_the_summary_aloud(self, flow: Flow):
        if not listen_enabled(settings):
            pytest.skip(
                "Listen spends ElevenLabs credits: set E2E_LISTEN=1 with ELEVENLABS_API_KEY,"
                " ELEVENLABS_VOICE_ID and ELEVENLABS_TTS_MODEL"
            )
        summary = need(flow.report, "the report").get("summary", "")
        assert len(summary) <= TTS_CHAR_BUDGET, (
            f"the summary has {len(summary)} characters, over the {TTS_CHAR_BUDGET} budget;"
            " not sent"
        )
        before = elevenlabs_characters(settings)
        url = f"/meetings/{flow.mid}/report/audio"
        started = time.monotonic()
        r = flow.client.get(url, headers=flow.alice)
        took = time.monotonic() - started
        audio = r.content if r.status_code == 200 else b""
        assert r.headers.get("content-type", "").startswith("audio/mpeg"), short(r)
        assert len(audio) > 1000 and (audio[:3] == b"ID3" or audio[0] == 0xFF), "not an MP3"
        started = time.monotonic()
        again = flow.client.get(url, headers=flow.alice)
        assert again.status_code == 200 and again.content == audio, (
            "a second click must serve the stored audio, with no new ElevenLabs call"
        )
        after = elevenlabs_characters(settings)
        used = after - before if before is not None and after is not None else None
        show(
            "listen",
            f"{len(audio)} bytes in {took:.1f}s for {len(summary)} characters; second click"
            f" {time.monotonic() - started:.2f}s; account characters used: {used}",
        )
        if used is not None:
            assert used <= TTS_CHAR_BUDGET, f"{used} characters spent, over the budget"

    def test_18_review_and_push_to_jira(self, flow: Flow):
        tasks = need(flow.report, "the report").get("tasks", [])
        assert tasks, "the write-up produced no task drafts"
        task = tasks[0]
        r = flow.client.patch(
            f"/meetings/{flow.mid}/tasks/{task['id']}",
            headers=flow.alice,
            json={**task, "title": task["title"] + " (reviewed)", "include": True},
        )
        assert r.status_code == 200 and r.json()["title"].endswith("(reviewed)"), short(r)

        r = flow.client.post(
            f"/meetings/{flow.mid}/tasks/push",
            headers=flow.alice,
            json={
                "task_ids": [task["id"]],
                "destination": "jira",
                "approved_by": flow.cast.alice.id,
            },
        )
        assert r.status_code == 200, short(r)
        pushed = r.json()
        key = pushed[0].get("key") if pushed else None
        assert key, f"the approved task did not become an issue: {pushed}"
        issue = flow.jira.read_back(key)
        owner = flow.cast.name(task.get("owner_id"))
        show(
            "push",
            f"{key} in the {flow.jira.kind} Jira: {issue.summary[:50] if issue else None!r}"
            f" assignee={issue.assignee if issue else None} owner={owner}"
            f" warning={pushed[0].get('warning')}",
        )
        assert issue and issue.summary.endswith("(reviewed)"), f"{key} not read back as reviewed"
        expected = owner if owner in flow.jira.accounts else None
        assert issue.assignee == expected, f"assigned to {issue.assignee}, expected {expected}"

    def test_19_decisions_and_open_tasks_pages(self, flow: Flow):
        need(flow.write_up_done, "the finished write-up")
        r = flow.client.get("/decisions", headers=flow.bob)
        assert r.status_code == 200 and r.json(), f"Decisions page: {short(r)}"
        r = flow.client.get("/tasks", headers=flow.bob, params={"open": True})
        assert r.status_code == 200, f"Tasks page: {short(r)}"
        show("pages", f"{len(r.json())} open tasks")

    def test_20_home_asks_about_the_past_meeting(self, flow: Flow):
        need(flow.write_up_done, "the finished write-up")
        r = flow.client.post(
            "/ask",
            headers=flow.carol,
            json={"question": HOME_QUESTION, "visibility": "public"},
        )
        assert r.status_code == 200, short(r)
        answer = flow.home_answer = r.json()
        show("home", f"sources={[s['label'] for s in answer['sources']]} {answer['text'][:160]!r}")
        assert any(
            s["kind"] == "meeting" and s.get("meeting_id") == flow.mid for s in answer["sources"]
        ), "the answer cites no source from this meeting"
        local_day = datetime.now(ZoneInfo(TEAM_TZ)).date()
        utc_day = datetime.now(UTC).date()
        if local_day != utc_day:  # the meeting is dated in the team's time zone, not UTC
            text = answer["text"]
            assert utc_day.isoformat() not in text and f"{utc_day:%B} {utc_day.day}" not in text, (
                f"dated {utc_day} (UTC), not {local_day} ({TEAM_TZ})"
            )

    def test_21_home_follow_up_uses_the_conversation(self, flow: Flow):
        first = need(flow.home_answer, "the first Home answer")
        r = flow.client.post(
            "/ask",
            headers=flow.carol,
            json={
                "question": "Who said that?",
                "visibility": "public",
                "history": [
                    {"role": "user", "text": HOME_QUESTION},
                    {"role": "agent", "text": first["text"]},
                ],
            },
        )
        assert r.status_code == 200, short(r)
        text = r.json()["text"]
        show("follow-up", repr(text[:160]))
        speaker = flow.cast.alice.name.split()[0]
        assert speaker in text, f"the follow-up does not name {speaker}"

    def test_22_connector_status(self, flow: Flow):
        r = flow.client.get("/settings/connectors", headers=flow.alice)
        assert r.status_code == 200, short(r)
        states = {c["name"]: c for c in r.json()}
        show("connectors", str(r.json()))
        assert states.get("jira", {}).get("state") == "connected", states.get("jira")
        github = states.get("github", {})
        # The fake GitHub lacks some read tools (list_pull_requests): failing, and saying why.
        assert github.get("state") == "connected" or (
            github.get("state") == "failing" and github.get("detail")
        ), github
