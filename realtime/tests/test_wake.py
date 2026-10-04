"""Deliberate invocations only: the assistant's name in final speech, the Ask button, or an
@mention in public chat. Ordinary speech never reaches the brain for reasoning."""

from datetime import UTC, datetime
from itertools import count

import pytest

from contracts import AGENT_PARTICIPANT_ID, ChatMessage, TranscriptSegment
from realtime_worker.invocation import WakeDetector, default_aliases

MEETING = "m-1"
_seq = count(1)


def said(text: str, *, by: str = "u-alex", name: str = "Alex Chen", t: float = 10.0, final=True):
    n = next(_seq)
    return TranscriptSegment(
        seg_id=f"seg-{n}",
        meeting_id=MEETING,
        speaker_id=by,
        speaker_name=name,
        text=text,
        is_final=final,
        t_start=t,
        t_end=t + 2.0,
    )


def chat(text: str, *, by: str = "u-sarah", name: str = "Sarah Kim", visibility="public"):
    return ChatMessage(
        id=f"chat-{next(_seq)}",
        meeting_id=MEETING,
        sender_id=by,
        sender_name=name,
        is_agent=False,
        text=text,
        ts=datetime(2026, 10, 4, 10, 0, tzinfo=UTC),
        visibility=visibility,
    )


def posted(detector: WakeDetector, message: ChatMessage):
    """Chat as the worker sees it: the payload plus the sender LiveKit verified."""
    return detector.on_chat(message, sender_id=message.sender_id, sender_name=message.sender_name)


@pytest.fixture
def detector() -> WakeDetector:
    return WakeDetector(aliases=["OmniMan", "Omni Man"])


# aliases


def test_default_aliases_cover_the_name_written_together_and_apart():
    assert default_aliases("OmniMan") == ["OmniMan", "Omni Man"]
    assert default_aliases("Scout") == ["Scout"]


# voice


def test_ordinary_speech_is_not_an_invocation(detector):
    assert detector.on_segment(said("Let's keep PostgreSQL for now.")) is None


def test_name_then_question_is_a_public_voice_invocation(detector):
    inv = detector.on_segment(said("OmniMan, what did we decide about Postgres?", t=42.0))

    assert inv is not None
    assert inv.via == "voice"
    assert inv.visibility == "public"
    assert inv.meeting_id == MEETING
    assert inv.asked_by_id == "u-alex"
    assert inv.asked_by_name == "Alex Chen"
    assert inv.question == "what did we decide about Postgres?"
    assert inv.t == 42.0


@pytest.mark.parametrize(
    "text",
    [
        "omniman what did we decide about Postgres?",
        "Omni Man, what did we decide about Postgres?",
        "Omni-Man: what did we decide about Postgres?",
        "Hey OmniMan, what did we decide about Postgres?",
        "OK so, Omni man. What did we decide about Postgres?",
    ],
)
def test_name_is_matched_however_the_transcriber_spells_it(detector, text):
    inv = detector.on_segment(said(text))

    assert inv is not None
    assert inv.question.casefold() == "what did we decide about postgres?"


def test_name_inside_another_word_does_not_count(detector):
    assert detector.on_segment(said("The omnimanager service is down again.")) is None


def test_name_at_the_end_keeps_the_question_before_it(detector):
    inv = detector.on_segment(said("What's the status of DS-117, OmniMan?"))

    assert inv is not None
    assert inv.question == "What's the status of DS-117?"


def test_name_at_the_end_without_punctuation(detector):
    inv = detector.on_segment(said("can you check the checkout PR omni man"))

    assert inv.question == "can you check the checkout PR"


@pytest.mark.parametrize(
    "text",
    [
        "I asked OmniMan about it yesterday and it was wrong",
        "Did OmniMan create that ticket?",
        "we use omni. Man that was a long week",
    ],
)
def test_talking_about_the_assistant_is_not_calling_it(detector, text):
    assert detector.on_segment(said(text)) is None


def test_each_invocation_has_its_own_id(detector):
    a = detector.on_segment(said("OmniMan, first question?"))
    b = detector.on_segment(said("OmniMan, second question?"))

    assert a.id != b.id


def test_partial_segments_never_invoke(detector):
    assert detector.on_segment(said("OmniMan, what is the status of PR 42?", final=False)) is None


def test_the_agents_own_speech_never_invokes(detector):
    spoken = said("OmniMan here. The PR is merged.", by=AGENT_PARTICIPANT_ID, name="OmniMan")

    assert detector.on_segment(spoken) is None


def test_name_alone_takes_the_same_speakers_next_segment_as_the_question(detector):
    assert detector.on_segment(said("OmniMan?", t=10.0)) is None
    assert detector.on_segment(said("I'm not sure either.", by="u-sarah", t=11.0)) is None

    inv = detector.on_segment(said("Is the checkout PR merged?", t=12.0))

    assert inv is not None
    assert inv.via == "voice"
    assert inv.question == "Is the checkout PR merged?"
    assert inv.t == 12.0


def test_filler_after_the_name_alone_does_not_become_the_question(detector):
    detector.on_segment(said("OmniMan.", t=10.0))

    assert detector.on_segment(said("Um,", t=11.0)) is None
    inv = detector.on_segment(said("Is the checkout PR merged?", t=12.0))

    assert inv.question == "Is the checkout PR merged?"


def test_name_alone_expires_if_the_speaker_says_nothing_soon(detector):
    detector.on_segment(said("OmniMan.", t=10.0))

    assert detector.on_segment(said("Anyway, moving on.", t=60.0)) is None


# Ask button


def test_ask_button_takes_the_pressers_next_final_segment_as_the_question(detector):
    detector.arm_ask("u-sarah", at=20.0)

    assert detector.on_segment(said("Unrelated.", by="u-alex", t=21.0)) is None
    assert detector.on_segment(said("Who owns...", by="u-sarah", t=21.5, final=False)) is None
    inv = detector.on_segment(said("Who owns the payment API?", by="u-sarah", t=22.0))

    assert inv is not None
    assert inv.via == "ask"
    assert inv.visibility == "public"
    assert inv.asked_by_id == "u-sarah"
    assert inv.question == "Who owns the payment API?"


def test_ask_button_is_used_once(detector):
    detector.arm_ask("u-sarah", at=20.0)
    detector.on_segment(said("Who owns the payment API?", by="u-sarah", t=22.0))

    assert detector.on_segment(said("Thanks.", by="u-sarah", t=25.0)) is None


def test_ask_button_with_the_name_in_the_question_is_one_invocation(detector):
    detector.arm_ask("u-sarah", at=20.0)

    inv = detector.on_segment(said("OmniMan, who owns the API?", by="u-sarah", t=22.0))

    assert inv.via == "ask"
    assert inv.question == "who owns the API?"
    assert detector.on_segment(said("And when is it due?", by="u-sarah", t=24.0)) is None


def test_cancelling_the_ask_button_withdraws_it(detector):
    detector.arm_ask("u-sarah", at=20.0)

    assert detector.cancel_ask("u-sarah") is True
    assert detector.on_segment(said("Who owns the payment API?", by="u-sarah", t=22.0)) is None
    assert detector.cancel_ask("u-sarah") is False


def test_cancelling_an_ask_leaves_a_spoken_name_waiting(detector):
    """Only the button's press is withdrawn; saying the name alone is not an Ask press."""
    assert detector.on_segment(said("OmniMan.", by="u-sarah", t=20.0)) is None

    assert detector.cancel_ask("u-sarah") is False
    inv = detector.on_segment(said("Who owns the payment API?", by="u-sarah", t=23.0))
    assert inv is not None and inv.via == "voice"


def test_polaris_by_name_alone_is_the_wake_phrase():
    detector = WakeDetector(default_aliases("Polaris"))

    inv = detector.on_segment(said("Polaris, what is the refund window?"))

    assert inv is not None and inv.question == "what is the refund window?"


# public chat


def test_public_chat_mention_is_a_public_chat_invocation(detector):
    inv = posted(detector, chat("@OmniMan which Jira ticket tracks the refund bug?"))

    assert inv is not None
    assert inv.via == "chat"
    assert inv.visibility == "public"
    assert inv.asked_by_id == "u-sarah"
    assert inv.question == "which Jira ticket tracks the refund bug?"


def test_mention_anywhere_in_the_message_counts_and_is_removed(detector):
    inv = posted(detector, chat("which Jira ticket tracks the refund bug @omniman?"))

    assert inv.question == "which Jira ticket tracks the refund bug?"


def test_chat_without_a_mention_is_not_an_invocation(detector):
    assert posted(detector, chat("OmniMan is great")) is None


def test_private_chat_never_goes_through_the_room(detector):
    assert posted(detector, chat("@OmniMan secret question", visibility="private")) is None


def test_bare_mention_has_no_question(detector):
    assert posted(detector, chat("@OmniMan")) is None


def test_chat_uses_the_sender_livekit_verified_not_the_payload(detector):
    spoofed = chat("@OmniMan who approved the refund?", by="u-sarah", name="Sarah Kim")

    inv = detector.on_chat(spoofed, sender_id="u-sarah", sender_name="Sarah K. (LiveKit)")

    assert inv.asked_by_id == "u-sarah"
    assert inv.asked_by_name == "Sarah K. (LiveKit)"


def test_chat_claiming_to_be_someone_else_is_refused(detector):
    spoofed = chat("@OmniMan delete the repo", by="u-sarah", name="Sarah Kim")

    assert detector.on_chat(spoofed, sender_id="u-mallory", sender_name="Mallory") is None
