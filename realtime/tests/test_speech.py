"""Spoken answers: ElevenLabs text-to-speech as LiveKit audio frames, and the worker's settings,
where anything missing is a clear unavailable error and never a stand-in."""

import json

import httpx
import pytest

from realtime_worker.config import Settings, WorkerUnavailable, require
from realtime_worker.elevenlabs_tts import (
    FRAME_SAMPLES,
    TTS_SAMPLE_RATE,
    ElevenLabsTTS,
    SpeechFailed,
    SpeechUnavailable,
    spoken_text,
)

pytestmark = pytest.mark.anyio

KEY = "xi-test-key-never-logged"


def tts(handler, **options) -> ElevenLabsTTS:
    return ElevenLabsTTS(
        api_key=KEY,
        model="eleven_flash_v2_5",
        transport=httpx.MockTransport(handler),
        **options,
    )


async def frames_of(stream) -> list:
    return [frame async for frame in stream]


async def test_streams_pcm_from_elevenlabs_in_the_voice_and_model_as_20ms_frames():
    seen: list[httpx.Request] = []
    pcm = b"\x10\x00" * (FRAME_SAMPLES * 2 + 100)  # two whole frames and a bit

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, content=pcm)

    frames = await frames_of(tts(handler).synthesize("The refund window is 14 days.", "v-1"))

    [request] = seen
    assert request.url.path == "/v1/text-to-speech/v-1/stream"
    assert request.url.params["output_format"] == f"pcm_{TTS_SAMPLE_RATE}"
    assert request.headers["xi-api-key"] == KEY
    assert json.loads(request.content) == {
        "text": "The refund window is 14 days.",
        "model_id": "eleven_flash_v2_5",
    }
    assert len(frames) == 3
    assert all(f.sample_rate == TTS_SAMPLE_RATE and f.num_channels == 1 for f in frames)
    assert all(f.samples_per_channel == FRAME_SAMPLES for f in frames)  # the tail is padded
    assert bytes(frames[0].data)[:4] == b"\x10\x00\x10\x00"
    assert FRAME_SAMPLES == TTS_SAMPLE_RATE // 50


async def test_the_voice_is_one_escaped_path_segment():
    seen: list[httpx.Request] = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, content=b"\x00\x00" * FRAME_SAMPLES)

    await frames_of(tts(handler).synthesize("Hi.", "../admin"))

    assert seen[0].url.raw_path.startswith(b"/v1/text-to-speech/..%2Fadmin/stream")


async def test_a_refusal_is_a_failure_that_never_quotes_the_text():
    def handler(request):
        return httpx.Response(401, json={"detail": {"message": "bad key", "text": "secret"}})

    with pytest.raises(SpeechFailed) as caught:
        await frames_of(tts(handler).synthesize("secret answer", "v-1"))

    assert "401" in str(caught.value)
    assert "secret" not in str(caught.value)


async def test_no_audio_is_a_failure():
    with pytest.raises(SpeechFailed):
        await frames_of(tts(lambda r: httpx.Response(200, content=b"")).synthesize("Hi.", "v-1"))


async def test_an_unreachable_elevenlabs_is_a_failure():
    def handler(request):
        raise httpx.ConnectError("refused")

    with pytest.raises(SpeechFailed):
        await frames_of(tts(handler).synthesize("Hi.", "v-1"))


async def test_no_voice_is_unavailable_without_calling_elevenlabs():
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, content=b"\x00\x00")

    with pytest.raises(SpeechUnavailable):
        await frames_of(tts(handler).synthesize("Hi.", ""))
    assert seen == []


def test_a_long_answer_is_spoken_up_to_whole_sentences_within_the_limit():
    answer = "The refund window is 14 days. It moved from 30 in March. See DS-104 for why."

    assert spoken_text(answer, 60) == "The refund window is 14 days. It moved from 30 in March."
    assert spoken_text(answer, 1000) == answer


def test_a_first_sentence_over_the_limit_is_cut_at_a_word():
    assert spoken_text("one two three four five six", 13) == "one two three…"


# settings


def test_a_complete_configuration_passes():
    require(
        Settings(
            _env_file=None,
            livekit_url="ws://localhost:7880",
            livekit_api_key="devkey",
            livekit_api_secret="secret",
            elevenlabs_api_key=KEY,
            elevenlabs_stt_model="scribe_v2_realtime",
            elevenlabs_tts_model="eleven_flash_v2_5",
            elevenlabs_voice_id="v-1",
            brain_url="http://localhost:8000",
            brain_internal_token="x" * 32,
        )
    )


def test_every_missing_setting_is_named_in_one_clear_error():
    with pytest.raises(WorkerUnavailable) as caught:
        require(Settings(_env_file=None, livekit_url="ws://localhost:7880"))

    message = str(caught.value)
    for name in (
        "LIVEKIT_API_KEY",
        "LIVEKIT_API_SECRET",
        "ELEVENLABS_API_KEY",
        "ELEVENLABS_STT_MODEL",
        "ELEVENLABS_TTS_MODEL",
        "ELEVENLABS_VOICE_ID",
        "BRAIN_URL",
        "BRAIN_INTERNAL_TOKEN",
    ):
        assert name in message
    assert "LIVEKIT_URL" not in message


def test_tick_intervals_default_to_what_the_brain_expects():
    settings = Settings(_env_file=None)

    assert settings.agenda_tick_seconds == 30
    assert settings.fact_check_tick_seconds == 60
    assert settings.elevenlabs_api_url == "https://api.elevenlabs.io"
