"""Deliberate invocations only: the assistant's name in final speech, the Ask button, or an
@mention in public chat. Ordinary speech never reaches the brain for reasoning."""

from datetime import UTC, datetime
from itertools import count

import pytest

from contracts import AGENT_PARTICIPANT_ID, ChatMessage, TranscriptSegment
from realtime_worker.invocation import TRAILING_SECONDS, WakeDetector, default_aliases

MEETING = "m-1"
_seq = count(1)


def said(
    text: str,
    *,
    by: str = "u-alex",
    name: str = "Alex Chen",
    t: float = 10.0,
    end: float | None = None,
    final=True,
):
    n = next(_seq)
    return TranscriptSegment(
        seg_id=f"seg-{n}",
        meeting_id=MEETING,
        speaker_id=by,
        speaker_name=name,
        text=text,
        is_final=final,
        t_start=t,
        t_end=t + 2.0 if end is None else end,
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


@pytest.mark.parametrize(
    "text",
    [
        "can you check the checkout PR omni man",
        "can you check the checkout PR OmniMan?",
        "can you check the checkout PR, OmniMan.",
        "I asked OmniMan.",
    ],
)
def test_name_at_the_end_needs_a_comma_before_it_and_a_question_mark(detector, text):
    assert detector.on_segment(said(text)) is None
    assert detector.due(float("inf")) == []


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


# a question that trails off


def test_a_question_that_trails_off_waits_for_the_rest_of_the_sentence(detector):
    """Scribe finalised a short pause as two segments: the question is both, joined."""
    assert detector.on_segment(said("OmniMan, what's the status of...", t=10.0)) is None

    inv = detector.on_segment(said("DS-104 in Jira.", t=14.0))

    assert inv is not None
    assert inv.via == "voice"
    assert inv.asked_by_id == "u-alex"
    assert inv.question == "what's the status of DS-104 in Jira."
    assert inv.t == 10.0
    assert detector.next_due() is None


@pytest.mark.parametrize(
    "text",
    [
        "OmniMan, what's the status of...",
        "OmniMan, what's the status of" + chr(0x2026),
        "OmniMan, what's the status of",
        "OmniMan, can you check the",
        "OmniMan, tell me about",
        "OmniMan, what's",
        "OmniMan, so what's",
        "OmniMan, check DS-104",
    ],
)
def test_unfinished_questions_are_held(detector, text):
    assert detector.on_segment(said(text, t=10.0)) is None
    assert detector.next_due() == pytest.approx(12.0 + TRAILING_SECONDS)


@pytest.mark.parametrize(
    "text, question",
    [
        ("OmniMan, what's the status of DS-104?", "what's the status of DS-104?"),
        ("OmniMan, any blockers?", "any blockers?"),
        ("OmniMan, summarise the last decision.", "summarise the last decision."),
    ],
)
def test_a_finished_question_is_sent_at_once(detector, text, question):
    inv = detector.on_segment(said(text))

    assert inv is not None and inv.question == question
    assert detector.next_due() is None


def test_a_held_question_is_sent_as_it_is_when_nothing_follows(detector):
    detector.on_segment(said("OmniMan, what's the status of...", t=10.0))

    assert detector.due(12.0 + TRAILING_SECONDS - 0.1) == []
    [inv] = detector.due(12.0 + TRAILING_SECONDS + 0.1)

    assert inv.via == "voice"
    assert inv.question == "what's the status of..."
    assert inv.t == 10.0
    assert detector.next_due() is None
    assert detector.on_segment(said("DS-104 in Jira.", t=19.0)) is None


def test_the_rest_said_too_late_is_not_joined_and_the_partial_still_goes(detector):
    detector.on_segment(said("OmniMan, what's the status of...", t=10.0))

    assert detector.on_segment(said("Anyway, moving on.", t=40.0)) is None
    [inv] = detector.due(40.0)
    assert inv.question == "what's the status of..."


def test_another_speaker_in_between_is_not_joined(detector):
    detector.on_segment(said("OmniMan, what's the status of...", t=10.0))

    assert detector.on_segment(said("I think it's blocked.", by="u-sarah", t=13.0)) is None
    inv = detector.on_segment(said("DS-104 in Jira.", t=15.0))

    assert inv is not None
    assert inv.asked_by_id == "u-alex"
    assert inv.question == "what's the status of DS-104 in Jira."


def test_the_name_alone_then_a_question_that_trails_off_waits_too(detector):
    detector.on_segment(said("OmniMan.", t=10.0))

    assert detector.on_segment(said("What's the status of...", t=12.0)) is None
    inv = detector.on_segment(said("DS-104?", t=15.0))

    assert inv.question == "What's the status of DS-104?"
    assert inv.t == 12.0


def test_the_ask_button_still_takes_the_next_segment_at_once(detector):
    detector.arm_ask("u-sarah", at=20.0)

    inv = detector.on_segment(said("Who owns the...", by="u-sarah", t=22.0))

    assert inv is not None and inv.via == "ask"
    assert inv.question == "Who owns the..."
    assert detector.next_due() is None


def test_a_chat_mention_that_trails_off_is_still_sent_at_once(detector):
    inv = posted(detector, chat("@OmniMan what's the status of"))

    assert inv is not None and inv.question == "what's the status of"


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


# real meetings (issue #108): times are seconds from the meeting start, as saved

DANIAL = {"by": "u-danial", "name": "Danial"}
REZA = {"by": "u-reza", "name": "Mohammad Reza"}
HOSSEIN = {"by": "u-hossein", "name": "Hossein"}


@pytest.fixture
def polaris() -> WakeDetector:
    return WakeDetector(default_aliases("Polaris"))


def replay(detector: WakeDetector, segments) -> list:
    """Segments in order, with the worker's timer: whatever is due by each segment's start goes
    first, and whatever is still held goes when the meeting ends."""
    invocations = []
    for segment in segments:
        invocations += detector.due(segment.t_start)
        if inv := detector.on_segment(segment):
            invocations.append(inv)
    invocations += detector.due(float("inf"))
    return invocations


FOUR_PERSON_MEETING = [
    said("I don't know what that might be.", t=25.8, end=27.0, **REZA),
    said("Okay. Sure.", t=27.3, end=28.1, **HOSSEIN),
    said(
        "خب، ولی یکم سرعتش چیزه. یکی حرف بزنه ببینیم میاد بالا. تو داری حرف می‌زنی. جا به جا.",
        t=33.1,
        end=39.0,
        **HOSSEIN,
    ),
    said("Hey, guys.", t=67.1, end=67.9, **HOSSEIN),
    said("Oh, I need-", t=71.2, end=72.0, **REZA),
    said("Um, Polaris,", t=112.5, end=114.8, **DANIAL),
    said("این دوباره. دوباره گفتی اون Polaris.", t=120.2, end=123.1, **DANIAL),
    said(
        "بذار ببینیم چیکار کنیم. بذار ببینیم چیکار کنیم. پسره دیگه.", t=125.2, end=130.0, **DANIAL
    ),
    said("یه سؤال این الان چرا فقط از...", t=137.9, end=140.5, **DANIAL),
    said("Polaris, how are you?", t=166.9, end=168.2, **DANIAL),
    said("Polaris,", t=219.5, end=220.3, **DANIAL),
    said("Polaris,", t=227.0, end=227.8, **DANIAL),
    said("Polaris,", t=248.7, end=249.5, **DANIAL),
    said("Uh, w- b- b- b- b", t=256.6, end=258.0, **REZA),
]


def test_the_four_person_meeting_asks_only_how_are_you(polaris):
    [inv] = replay(polaris, FOUR_PERSON_MEETING)

    assert inv.question == "how are you?"
    assert inv.asked_by_name == "Danial"
    assert inv.t == 166.9


def test_side_talk_that_mentions_the_name_after_the_name_alone_is_not_a_question(polaris):
    """1:52 "Um, Polaris," then 2:00 Persian side-talk, "you said Polaris again"."""
    assert replay(polaris, FOUR_PERSON_MEETING[5:7]) == []


def test_how_are_you_after_the_name_is_a_question(polaris):
    inv = polaris.on_segment(said("Polaris, how are you?", t=166.9, end=168.2, **DANIAL))

    assert inv is not None and inv.question == "how are you?"


def test_the_name_alone_three_times_asks_nothing(polaris):
    assert replay(polaris, FOUR_PERSON_MEETING[10:13]) == []


SOLO_MEETING = [
    said("Polaris, what's the status of...", t=274.0, end=276.3, **DANIAL),
    said("DS-104 in Jira.", t=277.8, end=279.4, **DANIAL),
    said(
        "Okay. Uh, we have decided to refund the affected users this week, and I'll email the "
        "affected users once",
        t=296.9,
        end=315.8,
        **DANIAL,
    ),
    said("the refunds are done.", t=316.0, end=316.9, **DANIAL),
]


def test_the_solo_meetings_question_split_by_a_pause_is_one_question(polaris):
    [inv] = replay(polaris, SOLO_MEETING)

    assert inv.question == "what's the status of DS-104 in Jira."
    assert inv.t == 274.0


def test_scribes_name_then_question_is_one_question(polaris):
    """The probe: Scribe commits the name said alone, and the question 2.4 s later."""
    [inv] = replay(
        polaris,
        [
            said("Polaris.", t=1.0, end=1.6, **DANIAL),
            said("Who owns DS-115?", t=4.0, end=5.2, **DANIAL),
        ],
    )

    assert inv.question == "Who owns DS-115?"
    assert inv.t == 4.0


def test_scribes_name_and_question_in_one_segment(polaris):
    inv = polaris.on_segment(said("Polaris, what is the status of DS-104 in Jira?", **DANIAL))

    assert inv is not None and inv.question == "what is the status of DS-104 in Jira?"


def test_another_speaker_after_the_name_alone_cancels_it(polaris):
    assert polaris.on_segment(said("Polaris.", t=10.0, end=10.6, **DANIAL)) is None
    assert polaris.on_segment(said("I don't know what that might be.", t=11.0, **REZA)) is None

    assert polaris.on_segment(said("Who owns DS-115?", t=13.5, **DANIAL)) is None
    assert polaris.due(float("inf")) == []


def test_another_speaker_cancels_a_fragment_waiting_after_the_name(polaris):
    polaris.on_segment(said("Polaris.", t=10.0, end=10.6, **DANIAL))
    assert polaris.on_segment(said("What's the status of...", t=11.0, end=12.5, **DANIAL)) is None

    assert polaris.on_segment(said("Hey, guys.", t=13.0, **HOSSEIN)) is None
    assert polaris.on_segment(said("DS-104?", t=15.0, **DANIAL)) is None
    assert polaris.due(float("inf")) == []


def test_the_name_alone_waits_eight_seconds(polaris):
    polaris.on_segment(said("Polaris.", t=10.0, end=10.6, **DANIAL))
    assert polaris.on_segment(said("Who owns DS-115?", t=18.8, **DANIAL)) is None

    polaris.on_segment(said("Polaris.", t=30.0, end=30.6, **DANIAL))
    inv = polaris.on_segment(said("Who owns DS-115?", t=38.4, **DANIAL))
    assert inv is not None and inv.question == "Who owns DS-115?"


def test_saying_the_name_alone_again_restarts_the_wait(polaris):
    polaris.on_segment(said("Polaris,", t=10.0, end=10.6, **DANIAL))
    polaris.on_segment(said("Polaris,", t=17.0, end=17.6, **DANIAL))

    inv = polaris.on_segment(said("Who owns DS-115?", t=24.0, **DANIAL))

    assert inv is not None and inv.question == "Who owns DS-115?"


@pytest.mark.parametrize(
    "fragment",
    [
        "Oh, I need-",
        "Uh, w- b- b- b- b",
        "یه سؤال این الان چرا فقط از...",
        "Okay. Sure.",
        "So we" + chr(0x2026),
    ],
)
def test_a_fragment_after_the_name_alone_is_never_the_question(polaris, fragment):
    polaris.on_segment(said("Polaris,", t=10.0, end=10.6, **DANIAL))

    assert polaris.on_segment(said(fragment, t=11.0, end=12.0, **DANIAL)) is None
    assert polaris.next_due() is not None  # it waits for the rest
    assert polaris.due(float("inf")) == []
    assert polaris.next_due() is None


def test_a_fragment_after_the_name_alone_joins_the_rest_of_the_question(polaris):
    polaris.on_segment(said("Polaris,", t=10.0, end=10.6, **DANIAL))
    polaris.on_segment(said("Okay. So", t=11.0, end=12.0, **DANIAL))

    inv = polaris.on_segment(said("who owns DS-115?", t=16.0, **DANIAL))

    assert inv is not None and inv.question == "Okay. So who owns DS-115?"
    assert inv.t == 11.0


@pytest.mark.parametrize(
    "text",
    [
        "Polaris said earlier that the release is on Friday.",
        "Polaris was wrong about the refund window" + chr(0x2026),
        "Polaris is down again",
        "Polaris answers in English only.",
        "I asked Polaris yesterday.",
        "این دوباره. دوباره گفتی اون Polaris.",
    ],
)
def test_talking_about_polaris_is_not_calling_it(polaris, text):
    assert polaris.on_segment(said(text, **DANIAL)) is None
    assert polaris.due(float("inf")) == []


@pytest.mark.parametrize(
    "text, question",
    [
        ("Polaris what's the status of DS-104?", "what's the status of DS-104?"),
        ("Polaris who owns DS-115?", "who owns DS-115?"),
        ("Polaris is DS-104 done?", "is DS-104 done?"),
        ("Polaris can you check DS-104?", "can you check DS-104?"),
        ("Polaris please summarize the last decision.", "please summarize the last decision."),
        ("Polaris: list the open blockers.", "list the open blockers."),
        ("Hey Polaris! Remind me what we decided.", "Remind me what we decided."),
        ("Polaris" + chr(0x2026) + " who owns DS-115?", "who owns DS-115?"),
        ("Polaris، وضعیت DS-104 چیه؟", "وضعیت DS-104 چیه؟"),
        ("What's the status of DS-117, Polaris?", "What's the status of DS-117?"),
        ("وضعیت DS-104 چیه، Polaris?", "وضعیت DS-104 چیه?"),
    ],
)
def test_calling_polaris_by_name(polaris, text, question):
    inv = polaris.on_segment(said(text, **DANIAL))

    assert inv is not None and inv.question == question


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
