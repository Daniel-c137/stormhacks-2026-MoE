"""The agent in one meeting, between the room and the brain.

Polaris reasons only when someone asks deliberately: by name, with the Ask button or with a public
@mention. A spoken question's answer is a shared card that stays silent until a participant
chooses Speak, Post in chat or Dismiss; a chat mention is answered in chat. On timers it asks the
brain to keep time against the agenda and to fact-check, and publishes what comes back, a private
fact-check only to its participant. Someone joining 5 minutes or more late, or back after 5 minutes
or more away, gets a private catch-up from the brain as a chat message only to them. People's
private chat never passes through here.
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
    Invocation,
    ResponseAction,
    ResponseCard,
    Topic,
    TranscriptSegment,
)
from contracts.agent import AgentStateName, ResponseCardStatus

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


class Transcription(Protocol):
    """The parts of TranscriptionManager the agent uses."""

    def arm_ask(self, participant_id: str) -> None: ...

    def cancel_ask(self, participant_id: str) -> bool: ...

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
        agenda_tick_seconds: float = 30,
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
        self._fact_check_tick_seconds = fact_check_tick_seconds
        self._ask_seconds = ask_seconds
        self._clock = clock
        self._catch_up_delay = catch_up_delay
        self.presence = Presence(catch_up_after)

        self._cards: dict[str, ResponseCard] = {}
        self._asking: dict[str, asyncio.Task[None]] = {}  # participant -> their Ask's expiry
        self._working = 0
        self._speaking = asyncio.Lock()
        self._last_agenda: str | None = None
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

    # answering

    async def on_invocation(self, invocation: Invocation) -> None:
        """A deliberate question from the room. Private questions go to the brain over HTTP from
        the board and never come through here."""
        if invocation.visibility != "public":
            log.warning("Ignored a %s invocation in the room", invocation.visibility)
            return
        self._stop_asking(invocation.asked_by_id)
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
        card = self._cards.get(action.card_id)
        if card is None or card.status != "pending":
            log.info("Ignored %s on card %s: not a pending card", action.action, action.card_id)
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
        # show_on_stage: the board publishes Topic.STAGE itself

    async def _speak(self, card: ResponseCard) -> None:
        if self._speaking.locked():
            log.info("Already speaking; ignored Speak on card %s", card.id)
            return
        failure = ""
        async with self._speaking:
            if card.status != "pending" or not self.state.can_move("speaking"):
                log.info("Not speaking card %s from %s", card.id, self.state.current.state)
                return
            text = spoken_text(card.answer.text, self._spoken_max_chars)
            voice = await self._voice()
            log.info("Speaking card %s: %d characters", card.id, len(text))
            await self._move("speaking", clip(f"Answering {card.invocation.asked_by_name}"))
            try:
                await self._speaker.play(self._tts.synthesize(text, voice or ""))
            except Exception as e:
                log.warning("Could not speak card %s: %s", card.id, e)
                failure = clip(f"Couldn't speak the answer: {e}")
        if not failure:
            await self._set_status(card, "spoken")
        await self._rest(failure)

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
        tracked = await self.brain.track_agenda(self.meeting_id)
        shown = tracked.agenda.model_dump_json(exclude={"generated_at"})
        if shown != self._last_agenda:
            await self.bus.publish(Topic.AGENDA, tracked.agenda)
            self._last_agenda = shown
        for nudge in tracked.nudges:
            await self.bus.publish(Topic.AGENDA_NUDGE, nudge)

    async def tick_fact_check(self) -> None:
        checked = await self.brain.fact_check(self.meeting_id)
        for check in checked.checks:
            if check.visibility == "public":
                await self.bus.publish(Topic.FACT_CHECK, check)
            elif check.recipient_id:
                await self.bus.publish(Topic.FACT_CHECK, check, to=[check.recipient_id])
            else:
                log.warning("Dropped private fact-check %s with no recipient", check.id)
        hand = checked.agent_state
        if hand and self.state.current.state == "idle" and self.state.can_move(hand.state):
            await self.state.move(hand)

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
        listening for an Ask, an answer waiting (hand raised), or idle."""
        if self._speaking.locked():
            return  # the answer being spoken settles the state when it ends
        if self._working:
            await self._move("working", self.state.current.detail)
        elif self._asking:
            await self._move("capturing", detail or "Listening for a question")
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
        cited = ", ".join(f"{s.label} ({s.url})" if s.url else s.label for s in answer.sources)
        lines.append(f"Sources: {cited}")
    if answer.unavailable:
        lines.append(f"Unavailable: {', '.join(answer.unavailable)}")
    return "\n\n".join(lines)


def clip(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= MAX_DETAIL else text[: MAX_DETAIL - 1] + "…"
