"""The agent in one meeting, between the room and the brain.

Polaris reasons only when someone asks deliberately: by name, with the Ask button or with a public
@mention. A spoken question's answer is a shared card that stays silent until a participant
chooses Speak, Post in chat or Dismiss; a chat mention is answered in chat. On timers it asks the
brain to keep time against the agenda and to fact-check; with Jev keeping time, it also checks
the agenda once each caption has settled in the brain. The agenda goes to the room; a fact-check
goes only to whoever made the claim, as a private chat message from the agent that is never
stored, spoken or shown to anyone else. Someone joining 2 minutes or more late, or back after 2
minutes or more away, gets a private catch-up from the brain the same way, only to them. Other
people's private chat never passes through here.
"""

import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Coroutine, Iterable
from datetime import UTC, datetime
from typing import Any, Protocol
from uuid import uuid4

from livekit import rtc

from contracts import (
    AGENT_PARTICIPANT_ID,
    AgentState,
    Answer,
    AskSignal,
    ChatMessage,
    FactCheck,
    Invocation,
    ResponseAction,
    ResponseCard,
    Source,
    Topic,
    TranscriptSegment,
)
from contracts.agent import AgentStateName, ResponseCardStatus, Verdict

from .brain_client import BrainClient
from .elevenlabs_tts import spoken_text
from .invocation import ASK_SECONDS, WakeDetector, default_aliases
from .presence import CATCH_UP_AFTER_SECONDS, Presence, Span
from .room import RoomBus
from .state import AgentStateMachine
from .tts import TextToSpeech

log = logging.getLogger(__name__)

MAX_DETAIL = 120
# A moment for a joiner's board to start listening before their catch-up is sent.
CATCH_UP_DELAY_S = 3.0
# Captions that settle within this of the first one waiting are checked against the agenda together.
AGENDA_GATHER_S = 0.5


# What a refused sender is told, privately. {agent} is the agent's name.
REFUSED: dict[str, str] = {
    "speak": "Only the host or an admin can let {agent} speak in this meeting.",
    "stop": "Only the host or an admin can stop {agent} in this meeting.",
    "send_to_chat": "Only the host or an admin can post {agent}'s answers in chat in this meeting.",
}
REFUSED_UNCHECKED = "I couldn't check who may do that just now, so I didn't. Try again in a moment."


class Transcription(Protocol):
    """The parts of TranscriptionManager the agent uses."""

    def arm_ask(self, participant_id: str) -> None: ...

    def cancel_ask(self, participant_id: str) -> bool: ...

    def cancel_listening(self, participant_id: str) -> bool: ...

    def recent_finals(self) -> list[TranscriptSegment]: ...


class Speaker(Protocol):
    """The agent's published audio track."""

    async def play(self, frames: AsyncIterator[rtc.AudioFrame]) -> None: ...


class Chat(Protocol):
    """Public room chat (LiveKit's lk.chat), sent as the agent. Returns the message id."""

    async def send(self, text: str) -> str: ...


class MeetingAgent:
    def __init__(
        self,
        meeting_id: str,
        *,
        bus: RoomBus,
        brain: BrainClient,
        state: AgentStateMachine,
        transcription: Transcription | None = None,
        tts: TextToSpeech,
        speaker: Speaker,
        chat: Chat,
        agent_name: str,
        default_voice_id: str | None,
        detector: WakeDetector | None = None,  # share the TranscriptionManager's
        spoken_max_chars: int = 600,
        agenda_tick_seconds: float = 10,
        agenda_after_captions: bool = False,
        agenda_check_delay: float = 3.5,
        fact_check_tick_seconds: float = 60,
        ask_seconds: float = ASK_SECONDS,
        clock: Callable[[], float] | None = None,
        catch_up_after: float = CATCH_UP_AFTER_SECONDS,
        catch_up_delay: float = CATCH_UP_DELAY_S,
    ):
        """transcription may be set after construction: it calls on_invocation. clock gives
        seconds since the meeting started; without one nobody is caught up."""
        self.meeting_id = meeting_id
        self.bus = bus
        self.brain = brain
        self.state = state
        self.transcription = transcription
        self._tts = tts
        self._speaker = speaker
        self._chat = chat
        self._agent_name = agent_name
        self._default_voice_id = default_voice_id
        self._detector = detector or WakeDetector(default_aliases(agent_name))
        self._spoken_max_chars = spoken_max_chars
        self._agenda_tick_seconds = agenda_tick_seconds
        self._agenda_after_captions = agenda_after_captions
        self._agenda_check_delay = agenda_check_delay
        self._fact_check_tick_seconds = fact_check_tick_seconds
        self._ask_seconds = ask_seconds
        self._clock = clock
        self._catch_up_delay = catch_up_delay
        self.presence = Presence(catch_up_after)

        self._cards: dict[str, ResponseCard] = {}
        self._asking: dict[str, asyncio.Task[None]] = {}  # participant -> their Ask's expiry
        self._listening: dict[str, str] = {}  # who is calling the agent by voice: id -> name
        self._working = 0
        self._speaking = asyncio.Lock()
        self._playback: asyncio.Task[None] | None = None  # the answer being spoken
        self._stop_requested = False
        self._last_agenda: str | None = None
        self._agenda_lock = asyncio.Lock()  # one agenda check at a time: a tick or a caption's
        self._agenda_due: list[float] = []  # when saved captions settle, on the meeting clock
        self._agenda_waiter: asyncio.Task[None] | None = None
        self._closed = False
        self._tasks: set[asyncio.Task[None]] = set()

    def start(self) -> None:
        """Called once the agent is in the room."""
        self.bus.subscribe(Topic.ASK, self.on_ask)
        self.bus.subscribe(Topic.RESPONSE_ACTION, self.on_action)
        self._spawn(self._say_joined())
        self._spawn(self._every(self._agenda_tick_seconds, self.tick_agenda, "Agenda"))
        self._spawn(self._every(self._fact_check_tick_seconds, self.tick_fact_check, "Fact-check"))

    async def _say_joined(self) -> None:
        """Lets the brain record that the agent attended. Failing only costs the report the
        agent's name among those present; the agent keeps working."""
        try:
            await self.brain.agent_joined(self.meeting_id)
        except Exception as e:
            log.warning("Could not tell the brain the agent joined %s: %s", self.meeting_id, e)

    async def aclose(self) -> None:
        self._closed = True  # the transcription flushes its last captions after this
        for task in [*self._tasks, *self._asking.values()]:
            task.cancel()
        await asyncio.gather(*self._tasks, *self._asking.values(), return_exceptions=True)
        self._tasks.clear()
        self._asking.clear()

    # the Ask button

    async def on_ask(self, signal: AskSignal, sender: str) -> None:
        if signal.by_id != sender:
            log.warning("Refused an Ask signal for %s sent by %s", signal.by_id, sender)
            return
        if self.transcription is None:
            return
        if signal.cancel:
            self.transcription.cancel_ask(sender)
            self.transcription.cancel_listening(sender)
            self._stop_asking(sender)
            if not self._asking and self.state.current.state == "capturing":
                await self._rest()
            return
        self.transcription.arm_ask(sender)
        self._stop_asking(sender)
        self._asking[sender] = asyncio.create_task(self._ask_expires(sender))
        await self._move("capturing", "Listening for a question")

    async def _ask_expires(self, participant_id: str) -> None:
        await asyncio.sleep(self._ask_seconds)
        self._asking.pop(participant_id, None)
        if not self._asking and self.state.current.state == "capturing":
            await self._rest()

    def _stop_asking(self, participant_id: str) -> None:
        if task := self._asking.pop(participant_id, None):
            task.cancel()

    # hearing its name

    async def on_listening(self, listening: dict[str, str]) -> None:
        """Who is calling the agent by voice (id -> name), from the transcription each time it
        changes: someone still saying "Polaris, ...", or waiting after the name alone or a
        question that trailed off. The agent shows it is listening to them."""
        self._listening = listening
        await self._rest()

    # answering

    async def on_invocation(self, invocation: Invocation) -> None:
        """A deliberate question from the room. Private questions go to the brain over HTTP from
        the board and never come through here."""
        if invocation.visibility != "public":
            log.warning("Ignored a %s invocation in the room", invocation.visibility)
            return
        self._stop_asking(invocation.asked_by_id)
        if invocation.via != "chat":  # a spoken question ends its asker's voice wait
            self._listening = {
                k: v for k, v in self._listening.items() if k != invocation.asked_by_id
            }
        self._working += 1
        try:
            await self._move("working", clip(f"Looking into: {invocation.question}"))
            recent = self.transcription.recent_finals() if self.transcription else []
            answer = await self.brain.invoke(invocation, recent)
        except Exception as e:
            log.warning("Invocation %s failed: %s", invocation.id, e)
            self._working -= 1
            await self._rest(clip(f"Couldn't answer: {e}"))
            return
        self._working -= 1
        if invocation.via == "chat":
            try:
                await self._post(answer)
            except Exception as e:
                log.warning("Could not post the answer to %s in chat: %s", invocation.id, e)
                await self._rest(clip(f"Couldn't post the answer in chat: {e}"))
                return
        else:
            card = ResponseCard(
                id=str(uuid4()), meeting_id=self.meeting_id, invocation=invocation, answer=answer
            )
            self._cards[card.id] = card
            await self.bus.publish(Topic.RESPONSE_CARD, card)
        await self._rest()

    # what people do with the card

    async def on_action(self, action: ResponseAction, sender: str) -> None:
        if action.by_id != sender:
            log.warning(
                "Refused a %s on card %s for %s sent by %s",
                action.action,
                action.card_id,
                action.by_id,
                sender,
            )
            return
        if action.action == "show_on_stage":
            return  # the board publishes Topic.STAGE itself
        if action.action == "stop":
            await self._stop(action, sender)
            return
        if self._pending(action) is None or not await self._may_act(action, sender):
            return
        card = self._pending(action)  # it may have changed while the brain was asked
        if card is None:
            return
        if action.action == "speak":
            await self._speak(card)
        elif action.action == "send_to_chat":
            await self._post(card.answer)
            await self._set_status(card, "sent_to_chat")
            await self._rest()
        elif action.action == "dismiss":
            await self._set_status(card, "dismissed")
            await self._rest()

    def _pending(self, action: ResponseAction) -> ResponseCard | None:
        card = self._cards.get(action.card_id)
        if card is None or card.status != "pending":
            log.info("Ignored %s on card %s: not a pending card", action.action, action.card_id)
            return None
        return card

    async def _may_act(self, action: ResponseAction, sender: str) -> bool:
        """Whether the team's who_can_allow lets the sender (the identity LiveKit verified) act
        on the card, asked of the brain each time, since the settings and who is an admin can
        change during the meeting. A refusal, or no answer from the brain, changes nothing; the
        sender is told privately, except for a Dismiss, which their board already hid for them.
        Logs never carry the card's text."""
        try:
            allowed = await self.brain.card_permission(self.meeting_id, sender)
        except Exception as e:
            log.warning(
                "Refused %s on card %s by %s: could not ask the brain who may act: %s",
                action.action,
                action.card_id,
                sender,
                e,
            )
            await self._tell_refused(sender, action, REFUSED_UNCHECKED)
            return False
        if not allowed:
            log.info(
                "Refused %s on card %s by %s: the team's settings do not let them",
                action.action,
                action.card_id,
                sender,
            )
            await self._tell_refused(sender, action, REFUSED.get(action.action, ""))
        return allowed

    async def _tell_refused(self, sender: str, action: ResponseAction, text: str) -> None:
        if action.action == "dismiss" or not text:
            return
        try:
            await self.send_private(sender, text.format(agent=self._agent_name))
        except Exception as e:
            log.warning("Could not tell %s their %s was refused: %s", sender, action.action, e)

    async def _speak(self, card: ResponseCard) -> None:
        """Plays the answer as its own task, so a Stop can cut it off (_stop). The card is
        `speaking` meanwhile, so the board shows Stop instead of Speak; it ends `spoken`, or back
        to `pending` when stopped or when the speech failed."""
        if self._speaking.locked():
            log.info("Already speaking; ignored Speak on card %s", card.id)
            return
        failure = ""
        stopped = False
        async with self._speaking:
            if card.status != "pending" or not self.state.can_move("speaking"):
                log.info("Not speaking card %s from %s", card.id, self.state.current.state)
                return
            text = spoken_text(card.answer.text, self._spoken_max_chars)
            voice = await self._voice()
            log.info("Speaking card %s: %d characters", card.id, len(text))
            await self._move("speaking", clip(f"Answering {card.invocation.asked_by_name}"))
            await self._set_status(card, "speaking")
            self._playback = asyncio.create_task(
                self._speaker.play(self._tts.synthesize(text, voice or ""))
            )
            self._stop_requested = False
            try:
                await self._playback
            except asyncio.CancelledError:
                current = asyncio.current_task()
                if not self._stop_requested or (current and current.cancelling()):
                    raise  # the agent itself is closing, not a Stop
                stopped = True
                log.info("Stopped speaking card %s", card.id)
            except Exception as e:
                log.warning("Could not speak card %s: %s", card.id, e)
                failure = clip(f"Couldn't speak the answer: {e}")
            finally:
                self._playback = None
        latest = self._cards.get(card.id, card)
        await self._set_status(latest, "pending" if stopped or failure else "spoken")
        await self._rest(failure)

    async def _stop(self, action: ResponseAction, sender: str) -> None:
        """Stop pressed on the card being spoken: the audio stops at once (TrackSpeaker clears
        what was queued). Anyone the team's who_can_allow lets act on cards may stop it."""
        card = self._cards.get(action.card_id)
        if card is None or card.status != "speaking" or self._playback is None:
            log.info("Ignored stop on card %s: it isn't being spoken", action.card_id)
            return
        if not await self._may_act(action, sender):
            return
        if self._playback is not None and not self._playback.done():
            self._stop_requested = True
            self._playback.cancel()

    async def _voice(self) -> str | None:
        """The team's chosen voice, else the default."""
        try:
            return (await self.brain.meeting(self.meeting_id)).voice_id or self._default_voice_id
        except Exception as e:
            log.warning("Could not read the team's voice, using the default: %s", e)
            return self._default_voice_id

    async def _set_status(self, card: ResponseCard, status: ResponseCardStatus) -> None:
        """Every card change goes to the room; the board replaces the card by id."""
        changed = card.model_copy(update={"status": status})
        self._cards[card.id] = changed
        await self.bus.publish(Topic.RESPONSE_CARD, changed)

    async def _post(self, answer: Answer) -> None:
        text = chat_text(answer)
        message_id = await self._chat.send(text)
        await self._save_chat(
            ChatMessage(
                id=message_id,
                meeting_id=self.meeting_id,
                sender_id=AGENT_PARTICIPANT_ID,
                sender_name=self._agent_name,
                is_agent=True,
                text=text,
                ts=datetime.now(UTC),
            )
        )

    # public chat

    async def on_chat(
        self, text: str, *, message_id: str, sender_id: str, sender_name: str, ts: datetime
    ) -> None:
        """A public room chat message; the sender is the identity LiveKit verified."""
        if sender_id == AGENT_PARTICIPANT_ID:
            return
        message = ChatMessage(
            id=message_id,
            meeting_id=self.meeting_id,
            sender_id=sender_id,
            sender_name=sender_name,
            is_agent=False,
            text=text,
            ts=ts,
        )
        await self._save_chat(message)
        detector = self._detector
        if detector and (
            invocation := detector.on_chat(message, sender_id=sender_id, sender_name=sender_name)
        ):
            await self.on_invocation(invocation)

    async def _save_chat(self, message: ChatMessage) -> None:
        try:
            await self.brain.ingest_public_chat(self.meeting_id, message)
        except Exception as e:
            log.warning("Could not save chat message %s: %s", message.id, e)

    # catching up late joiners

    def already_here(self, identities: Iterable[str]) -> None:
        """Who was in the room when the agent joined it: never caught up for how they got here,
        so a restarted worker does not catch up everyone again."""
        if self._clock:
            self.presence.already_here(identities, self._clock())

    def on_participant_joined(self, identity: str) -> None:
        """LiveKit's participant_connected."""
        if not self._clock:
            return
        if span := self.presence.joined(identity, self._clock()):
            self._spawn(self._catch_up(identity, span))

    def on_participant_left(self, identity: str) -> None:
        """LiveKit's participant_disconnected."""
        if self._clock:
            self.presence.left(identity, self._clock())

    async def _catch_up(self, participant_id: str, span: Span) -> None:
        """Asks the brain what they missed and sends it only to them; a failure sends nothing."""
        await asyncio.sleep(self._catch_up_delay)
        try:
            caught = await self.brain.catch_up(
                self.meeting_id, participant_id, span.since, span.until
            )
        except Exception as e:
            log.warning("Could not catch up %s in %s: %s", participant_id, self.meeting_id, e)
            return
        if not caught.text:
            log.info("Nothing to catch %s up on in %s", participant_id, self.meeting_id)
            return
        if not self.presence.present(participant_id):
            log.info("%s left %s before their catch-up was ready", participant_id, self.meeting_id)
            return
        try:
            await self.send_private(participant_id, caught.text)
        except Exception as e:
            log.warning("Could not send %s their catch-up: %s", participant_id, e)

    async def send_private(self, recipient_id: str, text: str) -> None:
        """A private chat message from the agent to one participant (Topic.PRIVATE_CHAT, only to
        them). Never broadcast, spoken or sent to the brain, so never stored."""
        message = ChatMessage(
            id=str(uuid4()),
            meeting_id=self.meeting_id,
            sender_id=AGENT_PARTICIPANT_ID,
            sender_name=self._agent_name,
            is_agent=True,
            text=text,
            ts=datetime.now(UTC),
            visibility="private",
            recipient_id=recipient_id,
        )
        await self.bus.publish(Topic.PRIVATE_CHAT, message, to=[recipient_id])

    # ticks

    async def tick_agenda(self) -> None:
        async with self._agenda_lock:
            tracked = await self.brain.track_agenda(self.meeting_id)
            shown = tracked.agenda.model_dump_json(exclude={"generated_at"})
            if shown != self._last_agenda:
                await self.bus.publish(Topic.AGENDA, tracked.agenda)
                self._last_agenda = shown
            for nudge in tracked.nudges:
                await self.bus.publish(Topic.AGENDA_NUDGE, nudge)

    def caption_saved(self, t_end: float) -> None:
        """A final caption that ended at `t_end` (meeting clock) reached the brain. With Jev
        keeping time the agenda is checked once it has settled there, agenda_check_delay after
        it ended; captions settling within AGENDA_GATHER_S of each other share one check."""
        if not self._agenda_after_captions or self._clock is None or self._closed:
            return
        self._agenda_due.append(t_end + self._agenda_check_delay)
        if self._agenda_waiter is None or self._agenda_waiter.done():
            self._agenda_waiter = asyncio.create_task(self._check_agenda_when_due())
            self._tasks.add(self._agenda_waiter)
            self._agenda_waiter.add_done_callback(self._tasks.discard)

    async def _check_agenda_when_due(self) -> None:
        assert self._clock is not None
        while self._agenda_due:
            first = min(self._agenda_due)
            due = max(d for d in self._agenda_due if d <= first + AGENDA_GATHER_S)
            await asyncio.sleep(max(0.0, due - self._clock()))
            self._agenda_due = [d for d in self._agenda_due if d > due]
            try:
                await self.tick_agenda()
            except Exception as e:
                log.warning("Agenda check failed; the next caption or tick tries again: %s", e)

    async def tick_fact_check(self) -> None:
        """Each check goes to whoever made the claim and nobody else, as a private chat message
        from the agent. It is never stored, spoken or broadcast."""
        checked = await self.brain.fact_check(self.meeting_id)
        for check in checked.checks:
            if not check.recipient_id:
                log.warning("Dropped fact-check %s: nobody to send it to", check.id)
                continue
            if (text := fact_check_text(check)) is not None:
                await self.send_private(check.recipient_id, text)

    async def _every(self, seconds: float, tick: Callable[[], Awaitable[None]], name: str):
        while True:
            await asyncio.sleep(seconds)
            try:
                await tick()
            except Exception as e:
                log.warning("%s tick failed; the next one tries again: %s", name, e)

    # state

    async def _move(self, to: AgentStateName, detail: str) -> None:
        if not self.state.can_move(to):
            log.info("Agent state stays %s (not %s)", self.state.current.state, to)
            return
        await self.state.move(AgentState(state=to, detail=detail))

    async def _rest(self, detail: str = "") -> None:
        """Where the agent settles when nothing is in progress: working on another question,
        listening for an Ask or to someone calling it, an answer waiting (hand raised), or
        idle."""
        if self._speaking.locked():
            return  # the answer being spoken settles the state when it ends
        if self._working:
            await self._move("working", self.state.current.detail)
        elif self._asking:
            await self._move("capturing", detail or "Listening for a question")
        elif self._listening:
            names = " and ".join(self._listening.values())
            await self._move("capturing", detail or clip(f"Listening to {names}"))
        elif any(c.status == "pending" for c in self._cards.values()):
            await self._move("hand_raised", detail or "Answer ready")
        else:
            await self._move("idle", detail)

    def _spawn(self, work: Coroutine[Any, Any, None]) -> None:
        task = asyncio.create_task(work)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)


def chat_text(answer: Answer) -> str:
    """The answer with its evidence, as Polaris posts it in public chat."""
    lines = [answer.text.strip()]
    if answer.sources:
        lines.append(f"Sources: {cited(answer.sources)}")
    if answer.unavailable:
        lines.append(f"Unavailable: {', '.join(answer.unavailable)}")
    return "\n\n".join(lines)


# How a fact-check opens, per verdict, and what it says when the brain gave no finding. A
# supported claim is not worded: nobody said anything wrong.
FACT_CHECK_LEADS: dict[Verdict, tuple[str, str]] = {
    "contradicted": ("The records disagree", "The records disagree with it"),
    "unknown": (
        "I couldn't confirm this",
        "I couldn't confirm this: the team's records don't settle it",
    ),
}


def fact_check_text(check: FactCheck) -> str | None:
    """The private message to whoever made the claim: what they said and when, what the records
    show, and the sources. None for a verdict that is not worth telling."""
    if check.verdict not in FACT_CHECK_LEADS:
        return None
    lead, bare = FACT_CHECK_LEADS[check.verdict]
    when = f" (at {clock(check.t)})" if check.t is not None else ""
    finding = " ".join(check.finding.split()).rstrip(".")
    records = f"{lead}: {finding}" if finding else bare
    said = f'You said "{check.claim}"{when}. {records}.'
    return f"{said}\nSources: {cited(check.sources)}" if check.sources else said


def cited(sources: list[Source]) -> str:
    return ", ".join(f"{s.label} ({s.url})" if s.url else s.label for s in sources)


def clock(t: float) -> str:
    """Seconds from the meeting start as mm:ss, or h:mm:ss after the first hour."""
    hours, rest = divmod(int(t), 3600)
    minutes, seconds = divmod(rest, 60)
    return f"{hours}:{minutes:02}:{seconds:02}" if hours else f"{minutes:02}:{seconds:02}"


def clip(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= MAX_DETAIL else text[: MAX_DETAIL - 1] + "…"
