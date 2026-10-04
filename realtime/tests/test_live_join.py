"""Polaris joins a real meeting: a LiveKit server, the brain on a seeded Postgres, the worker, and a
synthetic participant who says "Polaris, what is the refund window?" from a macOS `say` recording.

Checks that captions reach the room, the final segment is saved in the brain, the invocation
produces a shared response card, a Speak click makes Polaris publish audio, the agenda tick
publishes the agenda, and a public @mention in LiveKit chat is answered in chat with both messages
saved. Costs a few seconds of Scribe audio, two Gemini answers and at most SPOKEN_MAX characters of
ElevenLabs speech.

Run against the LiveKit dev server (`docker run --rm -p 7880:7880 -p 7881:7881 -p 7882:7882/udp
livekit/livekit-server --dev --bind 0.0.0.0`):

    LIVEKIT_URL=ws://localhost:7880 LIVEKIT_API_KEY=devkey LIVEKIT_API_SECRET=secret \
    ELEVENLABS_STT_MODEL=scribe_v2_realtime uv run pytest realtime -m live -k join -s
"""

import asyncio
import json
import os
import re
import secrets
import shutil
import socket
import subprocess
import sys
import time
import wave
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import numpy as np
import pytest
from livekit import api, rtc

from brain.config import Settings as BrainSettings
from contracts import (
    AGENT_PARTICIPANT_ID,
    Agenda,
    AgendaItem,
    Person,
    ResponseAction,
    ResponseCard,
    Team,
    Topic,
    TranscriptSegment,
)
from realtime_worker.config import Settings

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "brain" / "tests"))
from pg_support import fresh_database, pg_server  # noqa: E402, F401  (shared fixture)

pytestmark = [pytest.mark.live, pytest.mark.anyio]

QUESTION = "Polaris, what is the refund window?"
SPOKEN_MAX = 150  # characters of ElevenLabs speech this run may spend at most
ALEX = Person(id="u-alex", name="Alex Chen", short="Alex", initials="AC")
TEAM = Team(id="t-live", name="Checkout", member_ids=[])


def missing_settings() -> list[str]:
    settings = Settings()
    needed = {
        "LIVEKIT_URL": settings.livekit_url,
        "LIVEKIT_API_KEY": settings.livekit_api_key,
        "LIVEKIT_API_SECRET": settings.livekit_api_secret,
        "ELEVENLABS_API_KEY": settings.elevenlabs_api_key,
        "ELEVENLABS_STT_MODEL": settings.elevenlabs_stt_model,
        "ELEVENLABS_TTS_MODEL": settings.elevenlabs_tts_model,
        "ELEVENLABS_VOICE_ID": settings.elevenlabs_voice_id,
        "GEMINI_API_KEY": BrainSettings().gemini_api_key,
    }
    return [name for name, value in needed.items() if not value]


@pytest.fixture(scope="module", autouse=True)
def configured():
    if gaps := missing_settings():
        pytest.skip(f"needs {', '.join(gaps)}")
    if not shutil.which("say") or not shutil.which("afconvert"):
        pytest.skip("needs macOS say and afconvert to record the question")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def question_wav(tmp_path_factory) -> Path:
    folder = tmp_path_factory.mktemp("audio")
    aiff, wav = folder / "q.aiff", folder / "q.wav"
    subprocess.run(["say", "-o", str(aiff), QUESTION], check=True)
    subprocess.run(
        ["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", str(aiff), str(wav)],
        check=True,
    )
    return wav


@dataclass
class World:
    dsn: str
    meeting_id: str
    brain_url: str
    token: str
    logs: dict[str, Path] = field(default_factory=dict)


def seed(dsn: str) -> str:
    from brain.db import migrate
    from brain.pg_store import PostgresStore

    async def go() -> str:
        await migrate(dsn)
        store = PostgresStore(dsn)
        await store.create_team(TEAM)
        await store.upsert_person(ALEX, TEAM.id)
        meeting = await store.create_meeting(TEAM.id, "Refunds sync", host_id=ALEX.id)
        await store.add_participant(meeting.id, ALEX.id)
        await store.save_agenda(
            Agenda(
                meeting_id=meeting.id,
                person_id=ALEX.id,
                items=[AgendaItem(id="i-refunds", title="Refund window")],
                generated_at=meeting.started_at,
            )
        )
        return meeting.id

    return asyncio.run(go())


def wait_for_http(url: str, proc: subprocess.Popen, log: Path, what: str) -> None:
    deadline = time.monotonic() + 60
    while True:
        if proc.poll() is not None:
            pytest.fail(f"{what} exited with {proc.returncode}; see {log}")
        try:
            httpx.get(url, timeout=1)
            return
        except httpx.HTTPError:
            if time.monotonic() > deadline:
                pytest.fail(f"{what} did not start within 60 s; see {log}")
            time.sleep(0.3)


def stop(proc: subprocess.Popen) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        proc.kill()


@pytest.fixture(scope="module")
def world(pg_server, tmp_path_factory) -> Iterator[World]:  # noqa: F811
    token = secrets.token_urlsafe(32)
    logs = tmp_path_factory.mktemp("logs")
    with fresh_database(pg_server) as dsn:
        meeting_id = seed(dsn)
        port = free_port()
        brain_env = {
            **os.environ,
            "DATABASE_URL": dsn,
            "BRAIN_INTERNAL_TOKEN": token,
            "GITHUB_MCP_URL": "",
            "JIRA_MCP_URL": "",
        }
        brain_log = logs / "brain.log"
        with brain_log.open("w") as out:
            brain = subprocess.Popen(
                [sys.executable, "-m", "uvicorn", "brain.main:app", "--port", str(port)],
                cwd=REPO_ROOT,
                env=brain_env,
                stdout=out,
                stderr=subprocess.STDOUT,
            )
        url = f"http://127.0.0.1:{port}"
        try:
            wait_for_http(f"{url}/docs", brain, brain_log, "the brain")
            worker_env = {
                **os.environ,
                "BRAIN_URL": url,
                "BRAIN_INTERNAL_TOKEN": token,
                "SPOKEN_ANSWER_MAX_CHARS": str(SPOKEN_MAX),
                "AGENDA_TICK_SECONDS": "5",
                "FACT_CHECK_TICK_SECONDS": "5",
            }
            worker_log = logs / "worker.log"
            with worker_log.open("w") as out:
                worker = subprocess.Popen(
                    [sys.executable, "-m", "realtime_worker.worker", "start"],
                    cwd=REPO_ROOT,
                    env=worker_env,
                    stdout=out,
                    stderr=subprocess.STDOUT,
                )
            try:
                # Automatic dispatch only reaches workers registered when the room is created.
                deadline = time.monotonic() + 60
                while "registered worker" not in worker_log.read_text():
                    if worker.poll() is not None:
                        pytest.fail(f"the worker exited with {worker.returncode}; see {worker_log}")
                    if time.monotonic() > deadline:
                        pytest.fail(f"the worker did not register within 60 s; see {worker_log}")
                    time.sleep(0.3)
                yield World(dsn, meeting_id, url, token, {"brain": brain_log, "worker": worker_log})
            finally:
                stop(worker)
        finally:
            stop(brain)


def elevenlabs_characters() -> int | None:
    """Characters used this period, from GET /v1/user/subscription (free). Never prints the key."""
    settings = Settings()
    try:
        response = httpx.get(
            f"{settings.elevenlabs_api_url}/v1/user/subscription",
            headers={"xi-api-key": settings.elevenlabs_api_key or ""},
            timeout=15,
        )
    except httpx.HTTPError:
        return None
    if response.status_code != 200:
        return None
    return int(response.json().get("character_count", 0))


class Participant:
    """Alex in the meeting: publishes a microphone and records what the agent sends."""

    def __init__(self):
        self.room = rtc.Room()
        self.packets: list[tuple[float, str, dict]] = []
        self.agent_joined = asyncio.Event()
        self.agent_audio: list[tuple[float, int]] = []  # (time, peak) of each agent frame
        self._readers: list[asyncio.Task] = []
        self.chat: list[tuple[str, str]] = []  # (sender identity, text) on LiveKit chat
        self.source = rtc.AudioSource(16000, 1)

    async def join(self, meeting_id: str) -> None:
        settings = Settings()
        token = (
            api.AccessToken(settings.livekit_api_key, settings.livekit_api_secret)
            .with_identity(ALEX.id)
            .with_name(ALEX.name)
            .with_grants(api.VideoGrants(room_join=True, room=meeting_id, can_publish_data=True))
            .to_jwt()
        )

        @self.room.on("data_received")
        def on_data(packet: rtc.DataPacket) -> None:
            if packet.participant and packet.participant.identity == AGENT_PARTICIPANT_ID:
                self.packets.append((time.monotonic(), packet.topic, json.loads(packet.data)))

        @self.room.on("participant_connected")
        def on_join(participant: rtc.RemoteParticipant) -> None:
            if participant.identity == AGENT_PARTICIPANT_ID:
                self.agent_joined.set()

        @self.room.on("track_subscribed")
        def on_track(track, publication, participant) -> None:
            if (
                participant.identity == AGENT_PARTICIPANT_ID
                and track.kind == rtc.TrackKind.KIND_AUDIO
            ):
                self._readers.append(asyncio.create_task(self._listen(track)))

        def on_chat(reader: rtc.TextStreamReader, sender: str) -> None:
            async def read() -> None:
                self.chat.append((sender, await reader.read_all()))

            self._readers.append(asyncio.create_task(read()))

        self.room.register_text_stream_handler("lk.chat", on_chat)
        await self.room.connect(settings.livekit_url, token)
        if any(p == AGENT_PARTICIPANT_ID for p in self.room.remote_participants):
            self.agent_joined.set()

    async def _listen(self, track) -> None:
        async for event in rtc.AudioStream(track, sample_rate=24000, num_channels=1):
            peak = int(np.abs(np.frombuffer(event.frame.data, dtype=np.int16)).max(initial=0))
            self.agent_audio.append((time.monotonic(), peak))

    async def speak(self, wav: Path, silence_seconds: float) -> rtc.LocalTrackPublication:
        track = rtc.LocalAudioTrack.create_audio_track("mic", self.source)
        publication = await self.room.local_participant.publish_track(
            track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
        )
        await asyncio.sleep(1.0)  # the agent subscribes
        with wave.open(str(wav)) as w:
            pcm = w.readframes(w.getnframes())
        pcm += b"\x00\x00" * int(16000 * silence_seconds)
        per_frame = 160  # 10 ms
        for i in range(0, len(pcm) // 2, per_frame):
            chunk = pcm[i * 2 : (i + per_frame) * 2].ljust(per_frame * 2, b"\x00")
            await self.source.capture_frame(rtc.AudioFrame(chunk, 16000, 1, per_frame))
        return publication

    async def publish(self, topic: Topic, payload, to: list[str]) -> None:
        await self.room.local_participant.publish_data(
            payload.model_dump_json(), reliable=True, topic=topic.value, destination_identities=to
        )

    def on(self, topic: Topic, since: float = 0.0) -> list[dict]:
        return [p for t, name, p in self.packets if name == topic.value and t >= since]

    async def leave(self) -> None:
        for reader in self._readers:
            reader.cancel()
        await self.room.disconnect()


async def eventually(condition, timeout: float, what: str):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if result := condition():
            return result
        await asyncio.sleep(0.2)
    pytest.fail(f"timed out after {timeout:.0f} s waiting for {what}")


def spoken_characters(log: Path) -> int:
    return sum(int(n) for n in re.findall(r"Speaking card \S+: (\d+) characters", log.read_text()))


def scribe_seconds(log: Path) -> float:
    return sum(
        float(s) for s in re.findall(r"Scribe stream closed after ([\d.]+) s", log.read_text())
    )


async def test_polaris_joins_transcribes_answers_and_speaks_on_click(world, question_wav):
    from brain.pg_store import PostgresStore

    characters_before = elevenlabs_characters()
    alex = Participant()
    await alex.join(world.meeting_id)
    try:
        await asyncio.wait_for(alex.agent_joined.wait(), 30)
        agent = alex.room.remote_participants[AGENT_PARTICIPANT_ID]
        await eventually(lambda: agent.name == "Polaris", 5, "the agent's display name")

        mic = await alex.speak(question_wav, silence_seconds=2.5)

        def final_caption():
            finals = [p for p in alex.on(Topic.TRANSCRIPT) if p["is_final"]]
            return finals[0] if finals else None

        caption = TranscriptSegment.model_validate(
            await eventually(final_caption, 30, "a final caption")
        )
        partials = [p for p in alex.on(Topic.TRANSCRIPT) if not p["is_final"]]
        await alex.room.local_participant.unpublish_track(mic.sid)  # ends the Scribe stream
        assert caption.speaker_id == ALEX.id and caption.speaker_name == ALEX.name
        assert "refund" in caption.text.lower()

        store = PostgresStore(world.dsn)
        saved: list[TranscriptSegment] = []
        deadline = time.monotonic() + 15
        while not saved and time.monotonic() < deadline:
            saved = await store.transcript(world.meeting_id)
            await asyncio.sleep(0.3)
        assert saved and saved[0].seg_id == caption.seg_id

        card = ResponseCard.model_validate(
            (await eventually(lambda: alex.on(Topic.RESPONSE_CARD), 90, "a response card"))[0]
        )
        assert card.status == "pending"
        assert card.invocation.question.lower().startswith("what is the refund window")
        assert card.answer.text.strip()
        states = [p["state"] for p in alex.on(Topic.AGENT_STATE)]
        assert "working" in states and states[-1] == "hand_raised"
        # listening from the partial caption that said the name, before the question was final
        assert states.index("capturing") < states.index("working"), states
        listening = alex.on(Topic.AGENT_STATE)[states.index("capturing")]
        assert listening["detail"] == f"Listening to {ALEX.name}"
        assert not any(peak > 500 for _, peak in alex.agent_audio)  # silent until Speak

        clicked = time.monotonic()
        await alex.publish(
            Topic.RESPONSE_ACTION,
            ResponseAction(card_id=card.id, action="speak", by_id=ALEX.id),
            [AGENT_PARTICIPANT_ID],
        )
        spoken = await eventually(
            lambda: [c for c in alex.on(Topic.RESPONSE_CARD, clicked) if c["status"] == "spoken"],
            60,
            "the card to be spoken",
        )
        loud = [t for t, peak in alex.agent_audio if t >= clicked and peak > 500]
        after = [p["state"] for p in alex.on(Topic.AGENT_STATE, clicked)]
        assert loud, "Polaris published no audible audio after Speak"
        assert after[:1] == ["speaking"] and after[-1] == "idle"

        agenda = await eventually(lambda: alex.on(Topic.AGENDA), 15, "the agenda tick")
        assert agenda[-1]["items"][0]["title"] == "Refund window"

        # a public @mention in LiveKit chat is answered in chat, and both messages are saved
        await alex.room.local_participant.send_text(
            "@Polaris who owns the refund window?", topic="lk.chat"
        )
        [(_, reply)] = await eventually(
            lambda: [(who, text) for who, text in alex.chat if who == AGENT_PARTICIPANT_ID],
            90,
            "Polaris's chat answer",
        )
        chat = []
        deadline = time.monotonic() + 15
        while len(chat) < 2 and time.monotonic() < deadline:
            chat = await store.public_chat(world.meeting_id)
            await asyncio.sleep(0.3)
        assert [(m.sender_id, m.is_agent) for m in chat] == [
            (ALEX.id, False),
            (AGENT_PARTICIPANT_ID, True),
        ]
        assert chat[1].text == reply
    finally:
        await alex.leave()

    characters_after = elevenlabs_characters()
    print(
        "\n--- live join evidence ---"
        f"\ncaption partials: {len(partials)}; final: {caption.text!r} "
        f"t={caption.t_start:.2f}-{caption.t_end:.2f}"
        f"\nsaved segments: {[s.text for s in saved]}"
        f"\nstates before Speak: {states}; after: {after}"
        f"\ncard answer ({len(card.answer.text)} chars): {card.answer.text[:300]!r}"
        f"\nspoken card: {spoken[0]['id'] == card.id}; audible agent frames: {len(loud)}"
        f"\nagenda: {[i['title'] for i in agenda[-1]['items']]}"
        f"\nchat answer: {reply[:200]!r}; saved chat: {[m.sender_id for m in chat]}"
        f"\nScribe audio sent: {scribe_seconds(world.logs['worker']):.1f} s"
        f"\nElevenLabs speech sent: {spoken_characters(world.logs['worker'])} characters"
        f"\nElevenLabs characters used this period: before {characters_before}, "
        f"after {characters_after}"
        f"\nlogs: {world.logs}"
    )
