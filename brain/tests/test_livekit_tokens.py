from livekit.api import TokenVerifier

from brain.livekit_tokens import participant_token
from contracts import Meeting, Person

KEY = "test-key"
SECRET = "test-secret-that-is-long-enough-for-hs256"

MEETING = Meeting(
    id="m-1",
    team_id="t-1",
    title="Standup",
    status="live",
    code="abc",
    host_id="u-alex",
    participant_ids=[],
)
ALEX = Person(id="u-alex", name="Alex Chen", short="Alex", initials="AC")


def claims(token: str):
    return TokenVerifier(KEY, SECRET).verify(token)


def test_token_identity_is_the_account_id_and_name_is_the_display_name():
    c = claims(participant_token(MEETING, ALEX, is_host=False, api_key=KEY, api_secret=SECRET))

    assert c.identity == "u-alex"
    assert c.name == "Alex Chen"


def test_token_joins_the_meeting_room_and_can_publish_and_subscribe():
    c = claims(participant_token(MEETING, ALEX, is_host=False, api_key=KEY, api_secret=SECRET))

    assert c.video.room == "m-1"
    assert c.video.room_join is True
    assert c.video.can_publish is True
    assert c.video.can_subscribe is True
    assert c.video.can_publish_data is True


def test_only_the_host_gets_room_admin():
    host = claims(participant_token(MEETING, ALEX, is_host=True, api_key=KEY, api_secret=SECRET))
    member = claims(participant_token(MEETING, ALEX, is_host=False, api_key=KEY, api_secret=SECRET))

    assert host.video.room_admin is True
    assert not member.video.room_admin
