"""Which LiveKit tracks are transcribed: each participant's subscribed microphone, from when it is
heard until it is muted, unpublished or its participant leaves, and again after they come back."""

from dataclasses import dataclass

import pytest
from livekit import rtc

from contracts import AGENT_PARTICIPANT_ID
from realtime_worker.tracks import TrackRouter

pytestmark = pytest.mark.anyio


@dataclass
class Track:
    sid: str
    kind: int = rtc.TrackKind.KIND_AUDIO


@dataclass
class Publication:
    sid: str
    source: int = rtc.TrackSource.SOURCE_MICROPHONE
    muted: bool = False


@dataclass
class Participant:
    identity: str
    name: str = ""


class Audio:
    def __init__(self, track):
        self.track = track
        self.closed = False

    async def aclose(self):
        self.closed = True


class Transcription:
    def __init__(self):
        self.started: list[tuple[str, str, str, Audio]] = []
        self.stopped: list[str] = []
        self.left: list[str] = []

    def start(self, participant_id, participant_name, track_sid, audio) -> None:
        self.started.append((participant_id, participant_name, track_sid, audio))

    async def stop(self, track_sid) -> None:
        self.stopped.append(track_sid)

    async def stop_participant(self, participant_id) -> None:
        self.left.append(participant_id)


ALEX = Participant("u-alex", "Alex Chen")


@pytest.fixture
def transcription():
    return Transcription()


@pytest.fixture
def opened():
    return []


@pytest.fixture
def router(transcription, opened):
    def open_audio(track):
        audio = Audio(track)
        opened.append(audio)
        return audio

    return TrackRouter(transcription, open_audio)


def mic(sid: str = "TR_alex", **publication):
    return Track(sid), Publication(sid, **publication)


async def test_a_subscribed_microphone_is_transcribed_as_its_participant(router, transcription):
    track, publication = mic()

    router.subscribed(track, publication, ALEX)

    [(who, name, sid, audio)] = transcription.started
    assert (who, name, sid) == ("u-alex", "Alex Chen", "TR_alex")
    assert audio.track is track


async def test_a_participant_without_a_name_is_named_by_their_identity(router, transcription):
    router.subscribed(*mic(), Participant("u-alex", ""))

    assert transcription.started[0][1] == "u-alex"


@pytest.mark.parametrize(
    "track, publication, participant",
    [
        (Track("TR_v", kind=rtc.TrackKind.KIND_VIDEO), Publication("TR_v"), ALEX),
        (
            Track("TR_s"),
            Publication("TR_s", source=rtc.TrackSource.SOURCE_SCREENSHARE_AUDIO),
            ALEX,
        ),
        (Track("TR_a"), Publication("TR_a"), Participant(AGENT_PARTICIPANT_ID, "Polaris")),
    ],
)
async def test_other_tracks_and_the_agent_are_never_transcribed(
    router, transcription, opened, track, publication, participant
):
    router.subscribed(track, publication, participant)

    assert transcription.started == [] and opened == []


async def test_muting_stops_and_unmuting_starts_again_on_fresh_audio(router, transcription, opened):
    track, publication = mic()
    router.subscribed(track, publication, ALEX)

    await router.muted(ALEX, publication)
    router.unmuted(ALEX, publication)

    assert transcription.stopped == ["TR_alex"]
    assert opened[0].closed and not opened[1].closed
    assert [s[2] for s in transcription.started] == ["TR_alex", "TR_alex"]
    assert transcription.started[1][3] is opened[1]


async def test_a_track_muted_when_subscribed_waits_for_unmute(router, transcription):
    track, publication = mic(muted=True)

    router.subscribed(track, publication, ALEX)
    assert transcription.started == []

    router.unmuted(ALEX, publication)
    assert len(transcription.started) == 1


async def test_muting_twice_or_unmuting_twice_changes_nothing_more(router, transcription):
    track, publication = mic()
    router.subscribed(track, publication, ALEX)

    router.unmuted(ALEX, publication)
    await router.muted(ALEX, publication)
    await router.muted(ALEX, publication)

    assert len(transcription.started) == 1
    assert transcription.stopped == ["TR_alex"]


async def test_unsubscribing_stops_and_forgets_the_track(router, transcription, opened):
    track, publication = mic()
    router.subscribed(track, publication, ALEX)

    await router.unsubscribed(track, publication, ALEX)
    router.unmuted(ALEX, publication)

    assert transcription.stopped == ["TR_alex"]
    assert opened[0].closed
    assert len(transcription.started) == 1


async def test_leaving_stops_everything_of_that_participant_and_rejoining_starts_again(
    router, transcription, opened
):
    router.subscribed(*mic("TR_alex_1"), ALEX)

    await router.left(ALEX)
    router.subscribed(*mic("TR_alex_2"), ALEX)

    assert transcription.left == ["u-alex"]
    assert opened[0].closed
    assert [s[2] for s in transcription.started] == ["TR_alex_1", "TR_alex_2"]


async def test_closing_closes_every_open_audio_stream(router, opened):
    router.subscribed(*mic("TR_alex"), ALEX)
    router.subscribed(*mic("TR_sarah"), Participant("u-sarah", "Sarah Kim"))

    await router.aclose()

    assert all(a.closed for a in opened)
