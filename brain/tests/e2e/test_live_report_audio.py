"""Live: a short report summary read aloud by real ElevenLabs, through the brain's route. Needs
ELEVENLABS_API_KEY, ELEVENLABS_VOICE_ID and ELEVENLABS_TTS_MODEL. Never calls Gemini.
Deselected unless pytest runs with `-m live`."""

import asyncio

import pytest
from api_support import ALEX, create

from brain.config import Settings
from contracts import Report

live = Settings()

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not (live.elevenlabs_api_key and live.elevenlabs_voice_id and live.elevenlabs_tts_model),
        reason="set ELEVENLABS_API_KEY, ELEVENLABS_VOICE_ID and ELEVENLABS_TTS_MODEL to run"
        " against ElevenLabs",
    ),
]


def test_live_elevenlabs_reads_the_summary_once_then_it_is_served_from_the_store(
    client_as, store, settings
):
    settings.elevenlabs_api_key = live.elevenlabs_api_key
    settings.elevenlabs_api_url = live.elevenlabs_api_url
    settings.elevenlabs_voice_id = live.elevenlabs_voice_id
    settings.elevenlabs_tts_model = live.elevenlabs_tts_model
    meeting = create(client_as(ALEX))
    summary = "We agreed to ship the refund fix on Friday."

    async def write_up():
        await store.save_report(Report(meeting_id=meeting["id"], summary=summary))
        await store.set_status(meeting["id"], "needs_review")

    asyncio.run(write_up())

    first = client_as(ALEX).get(f"/meetings/{meeting['id']}/report/audio")
    assert first.status_code == 200, first.text
    assert first.headers["content-type"] == "audio/mpeg"
    audio = first.content
    print(f"{len(audio)} bytes of audio")
    assert len(audio) > 1000
    assert audio[:3] == b"ID3" or audio[0] == 0xFF  # an ID3 tag or an MPEG frame sync

    again = client_as(ALEX).get(f"/meetings/{meeting['id']}/report/audio")
    assert again.status_code == 200
    assert again.content == audio  # stored, not made again
