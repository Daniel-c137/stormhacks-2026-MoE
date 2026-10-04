"""Near-live English for speech in other languages (#106).

The room reads only English: a finished non-English sentence is translated and shown; one still
going after `provisional_seconds` is translated so far and shown as provisional (a partial),
again every `provisional_seconds` while it's open; when it finishes, the full translation
replaces it in place (same seg_id) and is what's saved. A provisional answer that comes back
after a newer one, or after the sentence finished, is dropped. If translation fails, the
original is shown and saved with its language and no translation.
"""

import asyncio
from collections.abc import Callable

import pytest

from contracts import Invocation, Topic, TranscriptSegment, TranslateResponse
from realtime_worker.brain_client import BrainUnavailable
from realtime_worker.invocation import WakeDetector
from realtime_worker.stt import SpeechPiece
from realtime_worker.transcription import TranscriptionManager

pytestmark = pytest.mark.anyio

MEETING = "m-1"
PERIOD = 0.05  # provisional_seconds in these tests
END = object()
SPANISH = "Por ahora nos quedamos con Postgres."
ENGLISH = "We keep Postgres for now."


class Mic:
    """A live microphone track: frames until the manager stops reading it."""

    def __aiter__(self):
        return self

    async def __anext__(self):
        await asyncio.sleep(0.002)
        return b""


class ScriptedSTT:
    """Yields the pieces each test pushes for its one speaker, as Scribe would."""

    def __init__(self):
        self.queue: asyncio.Queue = asyncio.Queue()
        self.t = 0.0

    def partial(self, text: str) -> None:
        self.queue.put_nowait(SpeechPiece(text, False, self.t, self.t + 0.5))

    def final(self, text: str, language: str | None) -> None:
        self.queue.put_nowait(SpeechPiece(text, True, self.t, self.t + 1.0, language))
        self.t += 2.0

    def end(self) -> None:
        self.queue.put_nowait(END)

    async def stream(self, audio):
        while (piece := await self.queue.get()) is not END:
            yield piece


class Translator:
    """The brain's translate call. Answers from `answers` (text -> (language, english)) or
    raises; `hold` keeps the answer for a text until released, to make a result arrive late."""

    def __init__(self):
        self.answers: dict[str, tuple[str, str] | Exception] = {}
        self.calls: list[tuple[str, str | None]] = []
        self.held: dict[str, asyncio.Event] = {}

    def hold(self, text: str) -> asyncio.Event:
        self.held[text] = asyncio.Event()
        return self.held[text]

    async def __call__(self, text: str, language: str | None) -> TranslateResponse:
        self.calls.append((text, language))
        if text in self.held:
            await self.held[text].wait()
        answer = self.answers.get(text)
        if answer is None:
            raise BrainUnavailable("no scripted translation")
        if isinstance(answer, Exception):
            raise answer
        detected, english = answer
        return TranslateResponse(language=detected, text=english)

    def texts(self) -> list[str]:
        return [text for text, _ in self.calls]


class Bus:
    def __init__(self):
        self.captions: list[TranscriptSegment] = []

    async def publish(self, topic, payload, *, to=None) -> None:
        if topic == Topic.TRANSCRIPT:
            self.captions.append(payload)

    def shown(self) -> list[str]:
        return [c.text for c in self.captions]


class Brain:
    def __init__(self):
        self.saved: list[TranscriptSegment] = []

    async def ingest_segments(self, meeting_id: str, segments: list[TranscriptSegment]) -> None:
        self.saved.extend(segments)


async def until(condition: Callable[[], bool], timeout: float = 2.0) -> None:
    async with asyncio.timeout(timeout):
        while not condition():
            await asyncio.sleep(0.005)


@pytest.fixture
def stt():
    return ScriptedSTT()


@pytest.fixture
def translator():
    return Translator()


@pytest.fixture
def bus():
    return Bus()


@pytest.fixture
def brain():
    return Brain()


@pytest.fixture
def invocations():
    return []


@pytest.fixture
async def manager(stt, translator, bus, brain, invocations):
    async def on_invocation(invocation: Invocation) -> None:
        invocations.append(invocation)

    m = TranscriptionManager(
        MEETING,
        stt=stt,
        bus=bus,
        brain=brain,
        detector=WakeDetector(["Polaris"]),
        on_invocation=on_invocation,
        translate=translator,
        provisional_seconds=PERIOD,
        translation_pause_seconds=60,
        clock=lambda: 0.0,
        drain_seconds=0.2,
    )
    m.start("u-lucia", "Lucía Gómez", "TR_lucia", Mic())
    yield m
    await m.aclose()


async def finish(manager: TranscriptionManager, stt: ScriptedSTT) -> None:
    stt.end()
    await manager.join()


# finished sentences


async def test_english_speech_is_shown_and_saved_as_today(manager, stt, translator, bus, brain):
    stt.partial("We keep")
    stt.final(ENGLISH, "en")
    await finish(manager, stt)

    assert bus.shown() == ["We keep", ENGLISH]
    assert [(s.text, s.language, s.original_text) for s in brain.saved] == [(ENGLISH, None, None)]
    assert translator.calls == []


async def test_a_finished_sentence_in_another_language_is_shown_and_saved_in_english(
    manager, stt, translator, bus, brain
):
    translator.answers[SPANISH] = ("es", ENGLISH)

    stt.final(SPANISH, "es")
    await finish(manager, stt)

    assert translator.calls == [(SPANISH, "es")]
    assert bus.shown() == [ENGLISH]
    [saved] = brain.saved
    assert (saved.text, saved.language, saved.original_text) == (ENGLISH, "es", SPANISH)


async def test_once_a_speaker_is_known_to_speak_another_language_the_original_is_hidden(
    manager, stt, translator, bus
):
    translator.answers[SPANISH] = ("es", ENGLISH)
    translator.answers["¿Y las migraciones?"] = ("es", "And the migrations?")
    stt.final(SPANISH, "es")
    await until(lambda: bus.shown() == [ENGLISH])

    stt.partial("¿Y las")
    stt.final("¿Y las migraciones?", "es")
    await finish(manager, stt)

    assert "¿Y las" not in bus.shown()
    assert bus.shown() == [ENGLISH, "And the migrations?"]


# sentences still going


async def test_a_sentence_still_going_is_translated_so_far_then_replaced_when_it_finishes(
    manager, stt, translator, bus, brain
):
    translator.answers[SPANISH] = ("es", ENGLISH)
    translator.answers["Por ahora nos quedamos"] = ("es", "For now we stay")
    stt.final(SPANISH, "es")  # Lucía is known to speak Spanish
    await until(lambda: len(brain.saved) == 1)

    stt.partial("Por ahora nos quedamos")
    await until(lambda: "For now we stay" in bus.shown())
    provisional = bus.captions[-1]
    stt.final(SPANISH, "es")
    await finish(manager, stt)

    assert (provisional.is_final, provisional.language) == (False, "es")
    assert provisional.original_text == "Por ahora nos quedamos"
    final = bus.captions[-1]
    assert final.seg_id == provisional.seg_id  # replaced in place
    assert (final.is_final, final.text) == (True, ENGLISH)
    assert [s.text for s in brain.saved] == [ENGLISH, ENGLISH]  # provisional never saved


async def test_an_open_sentence_is_retranslated_each_period_only_when_it_grew(
    manager, stt, translator, bus, brain
):
    translator.answers[SPANISH] = ("es", ENGLISH)
    translator.answers["Por ahora"] = ("es", "For now")
    translator.answers["Por ahora nos quedamos"] = ("es", "For now we stay")
    stt.final(SPANISH, "es")
    await until(lambda: len(brain.saved) == 1)

    stt.partial("Por ahora")
    await until(lambda: "For now" in bus.shown())
    await asyncio.sleep(PERIOD * 3)  # nothing new said: no new request
    assert translator.texts().count("Por ahora") == 1
    stt.partial("Por ahora nos quedamos")
    await until(lambda: "For now we stay" in bus.shown())
    stt.final(SPANISH, "es")
    await finish(manager, stt)

    assert translator.texts() == [SPANISH, "Por ahora", "Por ahora nos quedamos", SPANISH]


async def test_a_provisional_answer_arriving_after_the_sentence_finished_is_dropped(
    manager, stt, translator, bus, brain
):
    translator.answers[SPANISH] = ("es", ENGLISH)
    translator.answers["Por ahora nos"] = ("es", "For now we (late)")
    stt.final(SPANISH, "es")
    await until(lambda: len(brain.saved) == 1)
    late = translator.hold("Por ahora nos")

    stt.partial("Por ahora nos")
    await until(lambda: "Por ahora nos" in translator.texts())
    stt.final(SPANISH, "es")
    await until(lambda: len(brain.saved) == 2)
    late.set()
    await finish(manager, stt)

    assert "For now we (late)" not in bus.shown()
    assert bus.captions[-1].text == ENGLISH


async def test_a_new_speakers_first_words_show_until_their_language_is_known(
    manager, stt, translator, bus
):
    translator.answers["Hola a todos, hoy"] = ("es", "Hello everyone, today")
    translator.answers["Hola a todos, hoy hablamos"] = ("es", "Hello everyone, today we talk")

    stt.partial("Hola a todos, hoy")  # nobody knows Lucía's language yet
    await until(lambda: "Hello everyone, today" in bus.shown())
    stt.partial("Hola a todos, hoy hablamos")
    await until(lambda: "Hello everyone, today we talk" in bus.shown())
    await finish(manager, stt)

    assert bus.shown()[0] == "Hola a todos, hoy"  # before the first translation
    assert "Hola a todos, hoy hablamos" not in bus.shown()  # hidden once Spanish is known


async def test_a_new_speaker_found_to_speak_english_is_not_translated_again(
    manager, stt, translator, bus
):
    translator.answers["Okay so the plan"] = ("en", "Okay so the plan")

    stt.partial("Okay so the plan")
    await until(lambda: len(translator.calls) == 1)
    stt.final("Okay so the plan is to ship Friday.", "en")
    stt.partial("Next")
    await asyncio.sleep(PERIOD * 3)
    stt.final("Next item.", "en")
    await finish(manager, stt)

    assert translator.texts() == ["Okay so the plan"]
    assert bus.shown() == [
        "Okay so the plan",
        "Okay so the plan is to ship Friday.",
        "Next",
        "Next item.",
    ]


# failures and invocations


async def test_when_translation_fails_the_original_is_shown_and_saved_marked_untranslated(
    manager, stt, translator, bus, brain
):
    translator.answers[SPANISH] = BrainUnavailable("translation timed out")

    stt.final(SPANISH, "es")
    await finish(manager, stt)

    assert bus.shown() == [SPANISH]
    [saved] = brain.saved
    assert (saved.text, saved.language, saved.original_text) == (SPANISH, "es", None)


async def test_the_assistant_is_called_by_the_english_translation(
    manager, stt, translator, invocations
):
    translator.answers["Polaris, ¿qué decidimos sobre Postgres?"] = (
        "es",
        "Polaris, what did we decide about Postgres?",
    )

    stt.final("Polaris, ¿qué decidimos sobre Postgres?", "es")
    await finish(manager, stt)

    assert [i.question for i in invocations] == ["what did we decide about Postgres?"]


# failing translation keeps the room informed and the bill down (#106 review)


async def test_while_provisional_translation_fails_the_speakers_own_words_are_shown(
    manager, stt, translator, bus
):
    translator.answers[SPANISH] = ("es", ENGLISH)
    stt.final(SPANISH, "es")  # Lucía is known to speak Spanish, so her partials are hidden
    await until(lambda: bus.shown() == [ENGLISH])

    stt.partial("Por ahora")  # nothing scripted: the model is down
    await until(lambda: "Por ahora" in bus.shown())
    stt.partial("Por ahora nos quedamos")
    await until(lambda: "Por ahora nos quedamos" in bus.shown())
    stt.final(SPANISH, "es")
    await finish(manager, stt)

    assert bus.shown()[-1] == ENGLISH  # the finished sentence still gets its translation


async def test_a_failed_provisional_translation_pauses_the_next_ones(manager, stt, translator, bus):
    translator.answers[SPANISH] = ("es", ENGLISH)
    stt.final(SPANISH, "es")
    await until(lambda: bus.shown() == [ENGLISH])

    stt.partial("Por ahora")
    await until(lambda: "Por ahora" in translator.texts())
    for more in ("Por ahora nos", "Por ahora nos quedamos", "Por ahora nos quedamos con"):
        stt.partial(more)
        await asyncio.sleep(PERIOD * 2)
    stt.final(SPANISH, "es")
    await finish(manager, stt)

    assert translator.texts() == [SPANISH, "Por ahora", SPANISH]


async def test_no_model_configured_turns_translation_off_for_the_meeting(
    manager, stt, translator, bus, brain
):
    translator.answers[SPANISH] = BrainUnavailable("not configured", status=503)

    stt.final(SPANISH, "es")
    await until(lambda: len(brain.saved) == 1)
    stt.partial("¿Y las")
    await asyncio.sleep(PERIOD * 3)
    stt.final("¿Y las migraciones?", "es")
    await finish(manager, stt)

    assert translator.texts() == [SPANISH]  # asked once, never again this meeting
    assert bus.shown() == [SPANISH, "¿Y las", "¿Y las migraciones?"]
    assert [(s.text, s.language, s.original_text) for s in brain.saved] == [
        (SPANISH, "es", None),
        ("¿Y las migraciones?", "es", None),
    ]


# what the speaker's language is, and who decides it (#106 review)


async def test_provisional_translation_lets_the_model_detect_the_language(
    manager, stt, translator, bus
):
    translator.answers[SPANISH] = ("es", ENGLISH)
    translator.answers["Por ahora"] = ("es", "For now")
    stt.final(SPANISH, "es")
    await until(lambda: bus.shown() == [ENGLISH])

    stt.partial("Por ahora")
    await until(lambda: "For now" in bus.shown())
    await finish(manager, stt)

    assert translator.calls[:2] == [(SPANISH, "es"), ("Por ahora", None)]


async def test_a_short_final_in_another_language_does_not_change_the_speakers_language(
    manager, stt, translator, bus
):
    translator.answers[SPANISH] = ("es", ENGLISH)
    stt.final(SPANISH, "es")
    await until(lambda: bus.shown() == [ENGLISH])

    stt.final("Okay.", "en")  # one word: Lucía still speaks Spanish
    stt.partial("Entonces")
    await asyncio.sleep(PERIOD / 2)
    stt.final("Let's ship it on Friday.", "en")  # a real English sentence: she switched
    stt.partial("And then")
    await finish(manager, stt)

    assert "Entonces" not in bus.shown()
    assert "And then" in bus.shown()


async def test_the_assistant_is_called_in_persian_through_the_latin_spelling(
    manager, stt, translator, invocations
):
    said = "پولاریس، وضعیت DS-104 چیه؟"
    translator.answers[said] = ("fa", "Polaris, what's the status of DS-104?")

    stt.final(said, "fa")
    await finish(manager, stt)

    assert [i.question for i in invocations] == ["what's the status of DS-104?"]


async def test_a_persian_speaker_switching_to_english_mid_sentence_is_shown_as_said(
    manager, stt, translator, bus
):
    persian = "فعلاً با پستگرس می‌مونیم و بعد تصمیم می‌گیریم"
    translator.answers[persian] = ("fa", "We stay with Postgres for now and decide later")
    translator.answers["did you merge the PR"] = ("en", "did you merge the PR")
    stt.final(persian, "fa")  # Reza is known to speak Persian
    await until(lambda: len(bus.shown()) == 1)

    stt.partial("did you merge the PR")
    await until(lambda: "did you merge the PR" in bus.shown())
    stt.partial("did you merge the PR yet")
    await until(lambda: "did you merge the PR yet" in bus.shown())
    await finish(manager, stt)

    assert all(c.language in (None, "fa") for c in bus.captions)
    assert not any(c.language == "fa" and c.text.startswith("did you") for c in bus.captions)
