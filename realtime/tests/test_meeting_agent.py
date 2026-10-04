"""Polaris in one meeting: the Ask button, deliberate invocations to the brain, the shared answer
card and what people choose to do with it, public chat, and the timekeeping and fact-check ticks.
Polaris never speaks unless someone clicks Speak, and private things go only to their person: a
fact-check is a private chat message to whoever made the claim."""

import asyncio
import logging
import time
from datetime import UTC, datetime

import pytest

from contracts import (
    AGENT_PARTICIPANT_ID,
    Agenda,
    AgendaItem,
    AgendaNudge,
    AgendaTrackResponse,
    Answer,
    AskSignal,
    ChatMessage,
    FactCheck,
    FactCheckResponse,
    Invocation,
    Meeting,
    ResponseAction,
    ResponseCard,
    Source,
    Topic,
    TranscriptSegment,
    WorkerMeetingResponse,
)
from realtime_worker import meeting_agent
from realtime_worker.brain_client import BrainRejected, BrainUnavailable
from realtime_worker.elevenlabs_tts import SpeechFailed
from realtime_worker.meeting_agent import MeetingAgent
from realtime_worker.state import RoomAgentState

pytestmark = pytest.mark.anyio

MEETING = "m-1"
TS = datetime(2026, 10, 3, 17, 5, tzinfo=UTC)


class FakeBus:
    def __init__(self):
        self.published: list[tuple[Topic, object, list[str] | None]] = []
        self.handlers: dict[Topic, list] = {}

    async def publish(self, topic, payload, *, to=None) -> None:
        self.published.append((topic, payload, to))

    def subscribe(self, topic, handler) -> None:
        self.handlers.setdefault(topic, []).append(handler)

    async def deliver(self, topic: Topic, payload, sender: str) -> None:
        """A participant's packet, after the real bus checked the topic and payload type."""
        for handler in self.handlers.get(topic, []):
            await handler(payload, sender)

    def on(self, topic: Topic) -> list:
        return [(p, to) for t, p, to in self.published if t == topic]

    def states(self) -> list[str]:
        return [p.state for p, _ in self.on(Topic.AGENT_STATE)]

    def cards(self) -> list[ResponseCard]:
        return [p for p, _ in self.on(Topic.RESPONSE_CARD)]


class FakeBrain:
    def __init__(self):
        self.voice_id: str | None = "v-team"
        self.answer_text = "The refund window is 14 days."
        self.sources = [Source(kind="jira_issue", label="DS-104", url="https://jira/DS-104")]
        self.invoke_error: Exception | None = None
        self.meeting_error: Exception | None = None
        self.chat_error: Exception | None = None
        self.invoked: list[tuple[Invocation, list[TranscriptSegment]]] = []
        self.chat: list[ChatMessage] = []
        self.agenda_ticks: list = []  # scripted results: a response or an exception
        self.fact_ticks: list = []
        self.agenda_calls = 0
        self.fact_calls = 0
        self.joined: list[str] = []
        self.join_error: Exception | None = None
        self.allowed: set[str] | None = None  # who may act on a card; None is everyone
        self.permission_error: Exception | None = None
        self.permission_checks: list[tuple[str, str]] = []

    async def meeting(self, meeting_id: str) -> WorkerMeetingResponse:
        if self.meeting_error:
            raise self.meeting_error
        return WorkerMeetingResponse(
            meeting=Meeting(
                id=meeting_id,
                team_id="t-1",
                title="Standup",
                status="live",
                code="abc",
                host_id="u-alex",
                participant_ids=["u-alex", "u-sarah"],
            ),
            voice_id=self.voice_id,
        )

    async def card_permission(self, meeting_id: str, participant_id: str) -> bool:
        self.permission_checks.append((meeting_id, participant_id))
        if self.permission_error:
            raise self.permission_error
        return self.allowed is None or participant_id in self.allowed

    async def agent_joined(self, meeting_id: str) -> None:
        self.joined.append(meeting_id)
        if self.join_error:
            raise self.join_error

    async def invoke(self, invocation: Invocation, recent: list[TranscriptSegment]) -> Answer:
        self.invoked.append((invocation, recent))
        if self.invoke_error:
            raise self.invoke_error
        return Answer(
            id=f"a-{len(self.invoked)}",
            invocation_id=invocation.id,
            text=self.answer_text,
            sources=self.sources,
        )

    async def ingest_public_chat(self, meeting_id: str, message: ChatMessage) -> None:
        if self.chat_error:
            raise self.chat_error
        self.chat.append(message)

    async def track_agenda(self, meeting_id: str) -> AgendaTrackResponse:
        self.agenda_calls += 1
        result = self.agenda_ticks.pop(0) if self.agenda_ticks else agenda_response()
        if isinstance(result, Exception):
            raise result
        return result

    async def fact_check(self, meeting_id: str) -> FactCheckResponse:
        self.fact_calls += 1
        result = self.fact_ticks.pop(0) if self.fact_ticks else FactCheckResponse()
        if isinstance(result, Exception):
            raise result
        return result


class FakeTranscription:
    def __init__(self):
        self.armed: list[str] = []
        self.cancelled: list[str] = []
        self.stopped_listening: list[str] = []
        self.recent = [segment("Let's talk refunds.")]

    def arm_ask(self, participant_id: str) -> None:
        self.armed.append(participant_id)

    def cancel_ask(self, participant_id: str) -> bool:
        self.cancelled.append(participant_id)
        return participant_id in self.armed

    def cancel_listening(self, participant_id: str) -> bool:
        self.stopped_listening.append(participant_id)
        return True

    def recent_finals(self) -> list[TranscriptSegment]:
        return list(self.recent)


class FakeTTS:
    def __init__(self):
        self.calls: list[tuple[str, str]] = []
        self.error: Exception | None = None

    async def synthesize(self, text: str, voice_id: str):
        self.calls.append((text, voice_id))
        if self.error:
            raise self.error
        for n in range(3):
            yield f"frame-{n}"


class FakeSpeaker:
    def __init__(self):
        self.played: list[list] = []
        self.hold: asyncio.Event | None = None
        self.started = asyncio.Event()

    async def play(self, frames) -> None:
        self.started.set()
        got = [f async for f in frames]
        if self.hold:
            await self.hold.wait()
        self.played.append(got)


class FakeChat:
    def __init__(self):
        self.sent: list[str] = []

    async def send(self, text: str) -> str:
        self.sent.append(text)
        return f"lk-agent-{len(self.sent)}"


def segment(text: str, by: str = "u-alex") -> TranscriptSegment:
    return TranscriptSegment(
        seg_id=f"seg-{text}",
        meeting_id=MEETING,
        speaker_id=by,
        speaker_name="Alex Chen",
        text=text,
        is_final=True,
        t_start=10.0,
        t_end=12.0,
    )


def invocation(via="voice", by="u-alex", visibility="public") -> Invocation:
    return Invocation(
        id=f"inv-{via}",
        meeting_id=MEETING,
        via=via,
        visibility=visibility,
        asked_by_id=by,
        asked_by_name="Alex Chen",
        question="what is the refund window?",
        t=12.0,
    )


def agenda_response(*, current: str | None = None, nudges=()) -> AgendaTrackResponse:
    return AgendaTrackResponse(
        agenda=Agenda(
            meeting_id=MEETING,
            items=[AgendaItem(id="i-1", title="Refunds"), AgendaItem(id="i-2", title="Billing")],
            generated_at=datetime(2026, 10, 3, tzinfo=UTC),
            current_item_id=current,
        ),
        nudges=list(nudges),
    )


PR_41 = Source(kind="github_pr", label="dropsubs/app#41", url="https://github.com/d/a/pull/41")
RELEASE = Source(kind="github_release", label="dropsubs/app@v0.9.3")


def check(n: int, recipient_id: str | None = "u-sarah", **fields) -> FactCheck:
    defaults = {
        "claim": "PR 41 is released",
        "speaker_name": "Sarah Kim",
        "verdict": "contradicted",
        "confidence": 0.9,
        "severity": "high",
        "finding": "PR #41 was merged after the latest release, v0.9.3.",
        "sources": [PR_41, RELEASE],
        "t": 75.0,
    }
    return FactCheck(id=f"f-{n}", recipient_id=recipient_id, **(defaults | fields))


async def until(condition, timeout: float = 1.0) -> None:
    async with asyncio.timeout(timeout):
        while not condition():
            await asyncio.sleep(0.001)


@pytest.fixture
def bus():
    return FakeBus()


@pytest.fixture
def brain():
    return FakeBrain()


@pytest.fixture
def transcription():
    return FakeTranscription()


@pytest.fixture
def tts():
    return FakeTTS()


@pytest.fixture
def speaker():
    return FakeSpeaker()


@pytest.fixture
def chat():
    return FakeChat()


@pytest.fixture
async def make_agent(bus, brain, transcription, tts, speaker, chat):
    agents: list[MeetingAgent] = []

    def make(**options) -> MeetingAgent:
        agent = MeetingAgent(
            MEETING,
            bus=bus,
            brain=brain,
            state=RoomAgentState(bus),
            transcription=transcription,
            tts=tts,
            speaker=speaker,
            chat=chat,
            agent_name="Polaris",
            default_voice_id="v-default",
            **{"agenda_tick_seconds": 3600, "fact_check_tick_seconds": 3600, **options},
        )
        agent.start()
        agents.append(agent)
        return agent

    yield make
    for agent in agents:
        await agent.aclose()


@pytest.fixture
async def agent(make_agent):
    return make_agent()


async def card_for(agent: MeetingAgent, bus: FakeBus) -> ResponseCard:
    await agent.on_invocation(invocation())
    return bus.cards()[-1]


def act(card: ResponseCard, action: str, by: str = "u-sarah") -> ResponseAction:
    return ResponseAction(card_id=card.id, action=action, by_id=by)


# the Ask button


async def test_an_ask_press_arms_the_presser_and_shows_polaris_listening(agent, bus, transcription):
    await bus.deliver(Topic.ASK, AskSignal(by_id="u-alex"), "u-alex")

    assert transcription.armed == ["u-alex"]
    assert bus.states() == ["capturing"]
    assert all(to is None for _, to in bus.on(Topic.AGENT_STATE))


async def test_cancelling_an_ask_withdraws_it_and_goes_back_to_idle(agent, bus, transcription):
    await bus.deliver(Topic.ASK, AskSignal(by_id="u-alex"), "u-alex")
    await bus.deliver(Topic.ASK, AskSignal(by_id="u-alex", cancel=True), "u-alex")

    assert transcription.cancelled == ["u-alex"]
    assert bus.states() == ["capturing", "idle"]


async def test_polaris_keeps_listening_while_anyone_still_has_an_ask_pending(agent, bus):
    await bus.deliver(Topic.ASK, AskSignal(by_id="u-alex"), "u-alex")
    await bus.deliver(Topic.ASK, AskSignal(by_id="u-sarah"), "u-sarah")
    await bus.deliver(Topic.ASK, AskSignal(by_id="u-alex", cancel=True), "u-alex")

    assert bus.states()[-1] == "capturing"


async def test_an_ask_for_someone_else_is_refused(agent, bus, transcription, caplog):
    with caplog.at_level(logging.WARNING):
        await bus.deliver(Topic.ASK, AskSignal(by_id="u-alex"), "u-sarah")
        await bus.deliver(Topic.ASK, AskSignal(by_id="u-alex", cancel=True), "u-sarah")

    assert transcription.armed == []
    assert transcription.cancelled == []
    assert bus.states() == []
    assert "refus" in caplog.text.lower()


async def test_an_unanswered_ask_stops_listening_when_its_window_ends(make_agent, bus):
    make_agent(ask_seconds=0.02)

    await bus.deliver(Topic.ASK, AskSignal(by_id="u-alex"), "u-alex")
    await until(lambda: bus.states() == ["capturing", "idle"])


async def test_the_question_after_an_ask_moves_from_listening_to_working(agent, bus):
    await bus.deliver(Topic.ASK, AskSignal(by_id="u-alex"), "u-alex")

    await agent.on_invocation(invocation(via="ask"))

    assert bus.states() == ["capturing", "working", "hand_raised"]


# hearing its name: Polaris listens as soon as someone calls it


ALEX = {"u-alex": "Alex Chen"}
SARAH = {"u-sarah": "Sarah Kim"}


def details(bus: FakeBus) -> list[str]:
    return [p.detail for p, _ in bus.on(Topic.AGENT_STATE)]


async def test_hearing_its_name_shows_polaris_listening_to_the_speaker(agent, bus):
    await agent.on_listening(ALEX)

    assert bus.states() == ["capturing"]
    assert details(bus) == ["Listening to Alex Chen"]
    assert all(to is None for _, to in bus.on(Topic.AGENT_STATE))


async def test_listening_ends_when_nobody_is_calling_polaris_any_more(agent, bus):
    await agent.on_listening(ALEX)
    await agent.on_listening({})

    assert bus.states() == ["capturing", "idle"]


async def test_the_question_moves_listening_straight_to_working(agent, bus):
    """The worker hands on the question and the end of listening together: never idle between."""
    await agent.on_listening(ALEX)

    await asyncio.gather(agent.on_invocation(invocation()), agent.on_listening({}))

    assert bus.states() == ["capturing", "working", "hand_raised"]


async def test_listening_while_an_answer_waits_goes_back_to_the_raised_hand(agent, bus):
    await card_for(agent, bus)
    await agent.on_listening(SARAH)
    await agent.on_listening({})

    assert bus.states() == ["working", "hand_raised", "capturing", "hand_raised"]


async def test_someone_calling_polaris_while_it_works_is_listened_to_once_the_answer_is_in(
    agent, bus, brain
):
    answered = asyncio.Event()
    invoke = brain.invoke

    async def slow_invoke(inv, recent):
        await answered.wait()
        return await invoke(inv, recent)

    brain.invoke = slow_invoke
    working = asyncio.create_task(agent.on_invocation(invocation()))
    await until(lambda: bus.states() == ["working"])
    await agent.on_listening(SARAH)
    assert bus.states() == ["working"]

    answered.set()
    await working

    assert bus.states() == ["working", "capturing"]
    assert details(bus)[-1] == "Listening to Sarah Kim"


async def test_someone_calling_polaris_while_it_speaks_is_listened_to_once_it_stops(
    agent, bus, speaker
):
    card = await card_for(agent, bus)
    speaker.hold = asyncio.Event()
    speaking = asyncio.create_task(
        bus.deliver(Topic.RESPONSE_ACTION, act(card, "speak"), "u-sarah")
    )
    await until(lambda: bus.states()[-1] == "speaking")
    await agent.on_listening(ALEX)
    assert bus.states()[-1] == "speaking"

    speaker.hold.set()
    await speaking

    assert bus.states()[-1] == "capturing"


async def test_speak_works_while_polaris_listens_to_someone(agent, bus, tts):
    card = await card_for(agent, bus)
    await agent.on_listening(ALEX)

    await bus.deliver(Topic.RESPONSE_ACTION, act(card, "speak"), "u-sarah")

    assert len(tts.calls) == 1
    assert bus.states()[-3:] == ["capturing", "speaking", "capturing"]


async def test_a_question_asked_while_polaris_speaks_shows_working_not_speaking(
    agent, bus, brain, speaker
):
    card = await card_for(agent, bus)
    speaker.hold = asyncio.Event()
    speaking = asyncio.create_task(
        bus.deliver(Topic.RESPONSE_ACTION, act(card, "speak"), "u-sarah")
    )
    await until(lambda: bus.states()[-1] == "speaking")
    answered = asyncio.Event()
    invoke = brain.invoke

    async def slow_invoke(inv, recent):
        await answered.wait()
        return await invoke(inv, recent)

    brain.invoke = slow_invoke
    working = asyncio.create_task(agent.on_invocation(invocation()))
    await until(lambda: bus.states()[-1] == "working")
    speaker.hold.set()
    await speaking
    assert bus.states()[-1] == "working"

    answered.set()
    await working
    assert bus.states()[-1] == "hand_raised"


async def test_a_chat_question_leaves_polaris_listening_to_its_asker_by_voice(agent, bus):
    """The voice wait belongs to the transcription; only a spoken question ends it."""
    await agent.on_listening(ALEX)

    await agent.on_invocation(invocation(via="chat"))

    assert bus.states()[-1] == "capturing"


async def test_the_ask_button_and_a_voice_call_together_show_the_ask(agent, bus):
    await bus.deliver(Topic.ASK, AskSignal(by_id="u-sarah"), "u-sarah")
    await agent.on_listening(ALEX)

    assert details(bus)[-1] == "Listening for a question"


async def test_cancel_stops_listening_to_the_pressers_voice_too(agent, bus, transcription):
    await agent.on_listening(ALEX)

    await bus.deliver(Topic.ASK, AskSignal(by_id="u-alex", cancel=True), "u-alex")

    assert transcription.stopped_listening == ["u-alex"]


# checking the agenda once each caption has settled in the brain (with Jev on)


class MeetingClock:
    """Seconds since the meeting started, running in real time from 100 s."""

    def __init__(self):
        self.origin = time.monotonic() - 100

    def __call__(self) -> float:
        return time.monotonic() - self.origin


async def test_a_saved_caption_checks_the_agenda_once_it_has_settled(make_agent, bus, brain):
    clock = MeetingClock()
    agent = make_agent(agenda_after_captions=True, agenda_check_delay=0.1, clock=clock)

    agent.caption_saved(clock())
    await asyncio.sleep(0.02)
    assert brain.agenda_calls == 0  # not before it settled in the brain

    await until(lambda: brain.agenda_calls == 1)
    assert bus.on(Topic.AGENDA)  # published to the room like a timer tick


async def test_captions_that_settle_together_are_checked_once(make_agent, brain):
    clock = MeetingClock()
    agent = make_agent(agenda_after_captions=True, agenda_check_delay=0.05, clock=clock)

    now = clock()
    for t_end in (now - 0.02, now - 0.01, now):
        agent.caption_saved(t_end)
    await until(lambda: brain.agenda_calls == 1)
    await asyncio.sleep(0.1)

    assert brain.agenda_calls == 1


async def test_a_caption_that_settles_later_gets_its_own_check(make_agent, brain):
    clock = MeetingClock()
    agent = make_agent(agenda_after_captions=True, agenda_check_delay=0.05, clock=clock)

    agent.caption_saved(clock())
    agent.caption_saved(clock() + meeting_agent.AGENDA_GATHER_S + 0.1)

    await until(lambda: brain.agenda_calls == 1)
    await until(lambda: brain.agenda_calls == 2)


async def test_a_failing_check_is_logged_and_the_next_caption_checks_again(
    make_agent, brain, caplog
):
    clock = MeetingClock()
    agent = make_agent(agenda_after_captions=True, agenda_check_delay=0.01, clock=clock)
    brain.agenda_ticks = [BrainUnavailable("brain is down")]

    with caplog.at_level(logging.WARNING):
        agent.caption_saved(clock())
        await until(lambda: brain.agenda_calls == 1)
        agent.caption_saved(clock())
        await until(lambda: brain.agenda_calls == 2)

    assert "brain is down" in caplog.text


async def test_without_jev_a_saved_caption_never_checks_the_agenda(make_agent, brain):
    clock = MeetingClock()
    agent = make_agent(agenda_check_delay=0.01, clock=clock)

    agent.caption_saved(clock())
    await asyncio.sleep(0.1)

    assert brain.agenda_calls == 0


async def test_a_caption_check_and_the_timer_tick_never_overlap(make_agent, brain):
    clock = MeetingClock()
    agent = make_agent(
        agenda_after_captions=True, agenda_check_delay=0.01, agenda_tick_seconds=0.02, clock=clock
    )
    running, most = 0, 0
    track = brain.track_agenda

    async def slow_track(meeting_id):
        nonlocal running, most
        running += 1
        most = max(most, running)
        await asyncio.sleep(0.03)
        running -= 1
        return await track(meeting_id)

    brain.track_agenda = slow_track
    for _ in range(5):
        agent.caption_saved(clock())
        await asyncio.sleep(0.015)
    await until(lambda: brain.agenda_calls >= 4)

    assert most == 1


# answering


async def test_a_voice_question_becomes_a_pending_card_for_everyone_and_is_not_spoken(
    agent, bus, brain, tts, speaker
):
    await agent.on_invocation(invocation())

    [(asked, recent)] = brain.invoked
    assert asked.question == "what is the refund window?"
    assert [s.text for s in recent] == ["Let's talk refunds."]
    [(card, to)] = bus.on(Topic.RESPONSE_CARD)
    assert to is None
    assert card.meeting_id == MEETING
    assert card.status == "pending"
    assert card.invocation == asked
    assert card.answer.text == "The refund window is 14 days."
    assert bus.states() == ["working", "hand_raised"]
    assert bus.on(Topic.AGENT_STATE)[0][0].detail  # says what it is working on
    assert tts.calls == [] and speaker.played == []  # never speaks on its own


@pytest.mark.parametrize(
    "error", [BrainUnavailable("down"), BrainRejected(502, "Could not answer")]
)
async def test_a_failed_answer_goes_back_to_idle_and_says_so(agent, bus, brain, error):
    brain.invoke_error = error

    await agent.on_invocation(invocation())

    assert bus.cards() == []
    assert bus.states() == ["working", "idle"]
    failed = bus.on(Topic.AGENT_STATE)[-1][0]
    assert "couldn't answer" in failed.detail.lower()


async def test_a_public_chat_mention_is_answered_in_chat_not_with_a_card(agent, bus, brain, chat):
    await agent.on_invocation(invocation(via="chat"))

    assert bus.cards() == []
    [posted] = chat.sent
    assert posted.startswith("The refund window is 14 days.")
    assert "DS-104" in posted  # the evidence travels with the answer
    assert bus.states() == ["working", "idle"]
    [saved] = brain.chat
    assert (saved.sender_id, saved.is_agent, saved.text) == (AGENT_PARTICIPANT_ID, True, posted)
    assert saved.sender_name == "Polaris"


async def test_a_private_invocation_never_reaches_the_room(agent, bus, chat, brain):
    await agent.on_invocation(invocation(visibility="private", via="private"))

    assert brain.invoked == []
    assert bus.published == [] and chat.sent == []


# what people do with the card


async def test_speak_plays_the_answer_in_the_teams_voice_then_goes_idle(agent, bus, tts, speaker):
    card = await card_for(agent, bus)

    await bus.deliver(Topic.RESPONSE_ACTION, act(card, "speak"), "u-sarah")

    assert tts.calls == [("The refund window is 14 days.", "v-team")]
    assert speaker.played == [["frame-0", "frame-1", "frame-2"]]
    assert bus.states() == ["working", "hand_raised", "speaking", "idle"]
    assert bus.cards()[-1].id == card.id
    assert bus.cards()[-1].status == "spoken"


async def test_polaris_shows_speaking_while_the_audio_plays(agent, bus, speaker):
    card = await card_for(agent, bus)
    speaker.hold = asyncio.Event()

    task = asyncio.create_task(bus.deliver(Topic.RESPONSE_ACTION, act(card, "speak"), "u-sarah"))
    await speaker.started.wait()
    await asyncio.sleep(0)
    assert bus.states()[-1] == "speaking"
    speaker.hold.set()
    await task

    assert bus.states()[-1] == "idle"


@pytest.mark.parametrize("problem", ["no team voice", "brain down"])
async def test_speak_falls_back_to_the_default_voice(agent, bus, brain, tts, problem):
    if problem == "no team voice":
        brain.voice_id = None
    else:
        brain.meeting_error = BrainUnavailable("down")
    card = await card_for(agent, bus)

    await bus.deliver(Topic.RESPONSE_ACTION, act(card, "speak"), "u-sarah")

    assert tts.calls[0][1] == "v-default"


async def test_a_long_answer_is_spoken_only_up_to_the_limit(make_agent, bus, brain, tts):
    agent = make_agent(spoken_max_chars=40)
    brain.answer_text = "The refund window is 14 days. It moved from 30 days in March."
    card = await card_for(agent, bus)

    await bus.deliver(Topic.RESPONSE_ACTION, act(card, "speak"), "u-sarah")

    assert tts.calls[0][0] == "The refund window is 14 days."


async def test_two_speak_clicks_speak_once(agent, bus, tts, speaker):
    card = await card_for(agent, bus)
    speaker.hold = asyncio.Event()

    first = asyncio.create_task(bus.deliver(Topic.RESPONSE_ACTION, act(card, "speak"), "u-sarah"))
    await speaker.started.wait()
    await bus.deliver(Topic.RESPONSE_ACTION, act(card, "speak", by="u-alex"), "u-alex")
    speaker.hold.set()
    await first

    assert len(tts.calls) == 1


async def test_a_failed_speech_leaves_the_card_pending_and_says_so(agent, bus, tts, speaker):
    tts.error = SpeechFailed("ElevenLabs refused the speech request (401)")
    card = await card_for(agent, bus)

    await bus.deliver(Topic.RESPONSE_ACTION, act(card, "speak"), "u-sarah")

    assert speaker.played == []
    assert [c.status for c in bus.cards()] == ["pending"]
    assert bus.states() == ["working", "hand_raised", "speaking", "hand_raised"]
    assert "couldn't speak" in bus.on(Topic.AGENT_STATE)[-1][0].detail.lower()


async def test_post_in_chat_sends_the_answer_as_polaris_and_marks_the_card(
    agent, bus, chat, brain, tts
):
    card = await card_for(agent, bus)

    await bus.deliver(Topic.RESPONSE_ACTION, act(card, "send_to_chat"), "u-sarah")

    [posted] = chat.sent
    assert posted.startswith("The refund window is 14 days.")
    assert brain.chat[-1].is_agent and brain.chat[-1].text == posted
    assert bus.cards()[-1].status == "sent_to_chat"
    assert bus.states()[-1] == "idle"
    assert tts.calls == []


async def test_dismiss_closes_the_card_without_speaking_or_posting(agent, bus, chat, tts):
    card = await card_for(agent, bus)

    await bus.deliver(Topic.RESPONSE_ACTION, act(card, "dismiss"), "u-sarah")
    await bus.deliver(Topic.RESPONSE_ACTION, act(card, "speak"), "u-sarah")  # too late

    assert [c.status for c in bus.cards()] == ["pending", "dismissed"]
    assert chat.sent == [] and tts.calls == []
    assert bus.states()[-1] == "idle"


async def test_an_action_for_someone_else_or_an_unknown_card_is_refused(agent, bus, tts, chat):
    card = await card_for(agent, bus)

    await bus.deliver(Topic.RESPONSE_ACTION, act(card, "speak", by="u-alex"), "u-sarah")
    await bus.deliver(
        Topic.RESPONSE_ACTION,
        ResponseAction(card_id="nope", action="send_to_chat", by_id="u-sarah"),
        "u-sarah",
    )

    assert tts.calls == [] and chat.sent == []
    assert [c.status for c in bus.cards()] == ["pending"]


# who may act on the card (TeamSettings.who_can_allow, decided by the brain)


async def test_an_allowed_speak_is_spoken_after_asking_the_brain_about_the_sender(
    agent, bus, brain, tts, speaker
):
    brain.allowed = {"u-sarah"}
    card = await card_for(agent, bus)

    await bus.deliver(Topic.RESPONSE_ACTION, act(card, "speak"), "u-sarah")

    assert brain.permission_checks == [(MEETING, "u-sarah")]
    assert tts.calls and speaker.played
    assert bus.cards()[-1].status == "spoken"


@pytest.mark.parametrize("action", ["speak", "send_to_chat", "dismiss"])
async def test_a_refused_action_changes_nothing_and_is_logged_without_the_answer(
    agent, bus, brain, tts, speaker, chat, caplog, action
):
    brain.allowed = {"u-alex"}  # the host
    card = await card_for(agent, bus)
    states = bus.states()

    with caplog.at_level(logging.INFO, logger="realtime_worker"):
        await bus.deliver(Topic.RESPONSE_ACTION, act(card, action), "u-sarah")

    assert tts.calls == [] and speaker.played == [] and chat.sent == []
    assert [c.status for c in bus.cards()] == ["pending"]
    assert bus.states() == states
    refusals = [r for r in caplog.records if "refused" in r.getMessage().lower()]
    assert refusals and all(r.levelno == logging.INFO for r in refusals)
    assert "refund window" not in caplog.text.lower()


@pytest.mark.parametrize("action", ["speak", "send_to_chat"])
async def test_a_refused_sender_is_told_privately(agent, bus, brain, action):
    brain.allowed = {"u-alex"}
    card = await card_for(agent, bus)

    await bus.deliver(Topic.RESPONSE_ACTION, act(card, action), "u-sarah")

    [(message, to)] = bus.on(Topic.PRIVATE_CHAT)
    assert to == ["u-sarah"]
    assert message.recipient_id == "u-sarah" and message.visibility == "private"
    assert "host" in message.text.lower()
    assert "refund window" not in message.text.lower()


async def test_permission_is_decided_per_action_not_once(agent, bus, brain, tts):
    brain.allowed = {"u-alex"}
    card = await card_for(agent, bus)
    await bus.deliver(Topic.RESPONSE_ACTION, act(card, "speak"), "u-sarah")
    assert tts.calls == []

    brain.allowed = None  # the admin switched to everyone
    await bus.deliver(Topic.RESPONSE_ACTION, act(card, "speak"), "u-sarah")

    assert len(tts.calls) == 1
    assert brain.permission_checks == [(MEETING, "u-sarah"), (MEETING, "u-sarah")]


@pytest.mark.parametrize(
    "failure", [BrainUnavailable("down"), BrainRejected(404, "Meeting not found")]
)
async def test_speak_is_refused_when_the_brain_cannot_say_who_may_act(
    agent, bus, brain, tts, speaker, caplog, failure
):
    card = await card_for(agent, bus)
    brain.permission_error = failure

    with caplog.at_level(logging.WARNING, logger="realtime_worker"):
        await bus.deliver(Topic.RESPONSE_ACTION, act(card, "speak"), "u-sarah")

    assert tts.calls == [] and speaker.played == []
    assert [c.status for c in bus.cards()] == ["pending"]
    assert "u-sarah" in caplog.text
    assert "refund window" not in caplog.text.lower()


async def test_the_sender_checked_is_the_livekit_identity_not_the_payload(agent, bus, brain, tts):
    brain.allowed = {"u-alex"}
    card = await card_for(agent, bus)

    await bus.deliver(Topic.RESPONSE_ACTION, act(card, "speak", by="u-alex"), "u-sarah")

    assert tts.calls == []
    assert brain.permission_checks == []  # refused before asking: the payload names someone else


async def test_show_on_stage_is_left_to_the_board(agent, bus):
    card = await card_for(agent, bus)
    before = list(bus.published)

    await bus.deliver(Topic.RESPONSE_ACTION, act(card, "show_on_stage"), "u-sarah")

    assert bus.published == before


async def test_the_card_stays_while_a_new_question_is_answered(agent, bus):
    first = await card_for(agent, bus)
    await agent.on_invocation(invocation(by="u-sarah"))

    await bus.deliver(Topic.RESPONSE_ACTION, act(first, "speak"), "u-sarah")

    assert bus.states() == [
        "working",
        "hand_raised",
        "working",
        "hand_raised",
        "speaking",
        "hand_raised",  # the second answer is still waiting
    ]


# public chat


async def test_public_chat_goes_to_the_brain_with_the_livekit_sender(agent, brain):
    await agent.on_chat(
        "Shipping Friday.", message_id="lk-1", sender_id="u-alex", sender_name="Alex Chen", ts=TS
    )

    [saved] = brain.chat
    assert saved == ChatMessage(
        id="lk-1",
        meeting_id=MEETING,
        sender_id="u-alex",
        sender_name="Alex Chen",
        is_agent=False,
        text="Shipping Friday.",
        ts=TS,
        visibility="public",
    )


async def test_a_chat_mention_is_answered_for_the_verified_sender(agent, brain, chat):
    await agent.on_chat(
        "@Polaris what is the refund window?",
        message_id="lk-1",
        sender_id="u-sarah",
        sender_name="Sarah Kim",
        ts=TS,
    )

    [(asked, _)] = brain.invoked
    assert (asked.via, asked.asked_by_id, asked.question) == (
        "chat",
        "u-sarah",
        "what is the refund window?",
    )
    assert chat.sent and chat.sent[0].startswith("The refund window")


async def test_a_mention_is_answered_even_when_saving_the_chat_fails(agent, brain, chat):
    brain.chat_error = BrainUnavailable("down")

    await agent.on_chat(
        "@Polaris what is the refund window?",
        message_id="lk-1",
        sender_id="u-sarah",
        sender_name="Sarah Kim",
        ts=TS,
    )

    assert len(brain.invoked) == 1


async def test_the_agents_own_chat_is_not_forwarded_again(agent, brain):
    await agent.on_chat(
        "@Polaris hello",
        message_id="lk-1",
        sender_id=AGENT_PARTICIPANT_ID,
        sender_name="Polaris",
        ts=TS,
    )

    assert brain.chat == [] and brain.invoked == []


# joining


async def test_polaris_tells_the_brain_it_joined_once_it_starts(make_agent, brain):
    make_agent()

    await until(lambda: brain.joined)

    assert brain.joined == [MEETING]


@pytest.mark.parametrize(
    "failure", [BrainUnavailable("down"), BrainRejected(409, "The meeting is processing")]
)
async def test_a_failed_join_report_is_logged_and_polaris_keeps_working(
    make_agent, bus, brain, caplog, failure
):
    brain.join_error = failure

    with caplog.at_level(logging.WARNING):
        agent = make_agent(agenda_tick_seconds=0.01)
        await until(lambda: brain.joined and bus.on(Topic.AGENDA))
    await agent.on_invocation(invocation())

    assert "joined" in caplog.text
    assert brain.invoked and bus.cards()


# ticks


async def test_an_agenda_tick_publishes_the_agenda_and_each_nudge_to_everyone(agent, bus, brain):
    nudge = AgendaNudge(meeting_id=MEETING, item_id="i-2", text="Billing hasn't come up yet")
    brain.agenda_ticks = [agenda_response(current="i-1", nudges=[nudge])]

    await agent.tick_agenda()

    [(agenda, to)] = bus.on(Topic.AGENDA)
    assert (agenda.current_item_id, to) == ("i-1", None)
    assert bus.on(Topic.AGENDA_NUDGE) == [(nudge, None)]


async def test_an_unchanged_agenda_is_not_published_again(agent, bus, brain):
    brain.agenda_ticks = [agenda_response(current="i-1"), agenda_response(current="i-1")]

    await agent.tick_agenda()
    await agent.tick_agenda()

    assert len(bus.on(Topic.AGENDA)) == 1


async def test_a_contradicted_check_is_one_private_chat_message_to_the_claimant(
    agent, bus, brain, chat, tts
):
    brain.fact_ticks = [FactCheckResponse(checks=[check(1)])]

    await agent.tick_fact_check()

    [(message, to)] = bus.on(Topic.PRIVATE_CHAT)
    assert to == ["u-sarah"]
    assert message.text == (
        'You said "PR 41 is released" (at 01:15). The records disagree: PR #41 was merged after '
        "the latest release, v0.9.3.\n"
        "Sources: dropsubs/app#41 (https://github.com/d/a/pull/41), dropsubs/app@v0.9.3"
    )
    assert (message.meeting_id, message.recipient_id, message.visibility) == (
        MEETING,
        "u-sarah",
        "private",
    )
    assert (message.sender_id, message.sender_name, message.is_agent) == (
        AGENT_PARTICIPANT_ID,
        "Polaris",
        True,
    )
    assert [(topic, to) for topic, _, to in bus.published] == [
        (Topic.PRIVATE_CHAT, ["u-sarah"])
    ]  # nothing to the room, no raised hand
    assert brain.chat == [] and chat.sent == []  # never stored, never in public chat
    assert tts.calls == []  # never spoken


async def test_each_check_goes_only_to_its_own_claimant(agent, bus, brain):
    brain.fact_ticks = [
        FactCheckResponse(checks=[check(1), check(2, "u-alex", claim="DS-104 is closed")])
    ]

    await agent.tick_fact_check()

    sent = [(m.recipient_id, to, m.text.split(" (at")[0]) for m, to in bus.on(Topic.PRIVATE_CHAT)]
    assert sent == [
        ("u-sarah", ["u-sarah"], 'You said "PR 41 is released"'),
        ("u-alex", ["u-alex"], 'You said "DS-104 is closed"'),
    ]


async def test_an_unverified_check_says_it_could_not_be_confirmed(agent, bus, brain):
    brain.fact_ticks = [
        FactCheckResponse(
            checks=[
                check(1, verdict="unknown", finding="No release lists PR 41.", sources=[RELEASE]),
                check(2, verdict="unknown", finding="", sources=[], t=3725.0),
            ]
        )
    ]

    await agent.tick_fact_check()

    assert [m.text for m, _ in bus.on(Topic.PRIVATE_CHAT)] == [
        'You said "PR 41 is released" (at 01:15). I couldn\'t confirm this: No release lists PR '
        "41.\nSources: dropsubs/app@v0.9.3",
        "You said \"PR 41 is released\" (at 1:02:05). I couldn't confirm this: the team's records "
        "don't settle it.",
    ]


async def test_a_check_with_no_claimant_is_dropped(agent, bus, brain, caplog):
    brain.fact_ticks = [FactCheckResponse(checks=[check(1, recipient_id=None)])]

    with caplog.at_level(logging.WARNING):
        await agent.tick_fact_check()

    assert bus.published == []
    assert "f-1" in caplog.text and "PR 41" not in caplog.text


def test_only_what_someone_got_wrong_or_unconfirmed_is_worded():
    assert meeting_agent.fact_check_text(check(1, verdict="supported")) is None
    contradicted = meeting_agent.fact_check_text(check(1, finding="", sources=[], t=None))
    assert contradicted == 'You said "PR 41 is released". The records disagree with it.'


async def test_a_fact_check_never_changes_what_polaris_is_doing(agent, bus, brain, speaker):
    card = await card_for(agent, bus)
    speaker.hold = asyncio.Event()
    task = asyncio.create_task(bus.deliver(Topic.RESPONSE_ACTION, act(card, "speak"), "u-sarah"))
    await speaker.started.wait()
    brain.fact_ticks = [FactCheckResponse(checks=[check(1)])]

    await agent.tick_fact_check()
    speaker.hold.set()
    await task

    assert bus.states() == ["working", "hand_raised", "speaking", "idle"]
    assert [to for _, to in bus.on(Topic.PRIVATE_CHAT)] == [["u-sarah"]]


async def test_a_failing_tick_is_logged_and_the_next_one_tries_again(
    make_agent, bus, brain, caplog
):
    brain.agenda_ticks = [BrainUnavailable("down"), agenda_response(current="i-2")]
    brain.fact_ticks = [
        BrainRejected(409, "Only a live meeting is fact-checked"),
        FactCheckResponse(checks=[check(1)]),
    ]

    with caplog.at_level(logging.WARNING):
        make_agent(agenda_tick_seconds=0.01, fact_check_tick_seconds=0.01)
        await until(lambda: bus.on(Topic.AGENDA) and bus.on(Topic.PRIVATE_CHAT))

    assert brain.agenda_calls >= 2 and brain.fact_calls >= 2
    assert "tick" in caplog.text.lower()


async def test_ticks_stop_when_the_agent_closes(make_agent, brain):
    agent = make_agent(agenda_tick_seconds=0.01, fact_check_tick_seconds=0.01)
    await until(lambda: brain.agenda_calls >= 1)

    await agent.aclose()
    calls = brain.agenda_calls
    await asyncio.sleep(0.05)

    assert brain.agenda_calls == calls
