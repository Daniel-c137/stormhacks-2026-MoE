from datetime import timedelta

from livekit.api import AccessToken, VideoGrants

from contracts import Meeting, Person


def participant_token(
    meeting: Meeting,
    person: Person,
    *,
    is_host: bool,
    api_key: str,
    api_secret: str,
    ttl: timedelta | None = None,
) -> str:
    """LiveKit access token; identity is the account id so captions map to the right person."""
    grants = VideoGrants(
        room_join=True,
        room=meeting.id,
        room_admin=is_host or None,
        can_publish=True,
        can_subscribe=True,
        can_publish_data=True,
        can_update_own_metadata=True,  # the board raises a hand with the participant's attributes
    )
    token = (
        AccessToken(api_key, api_secret)
        .with_identity(person.id)
        .with_name(person.name)
        .with_grants(grants)
    )
    if ttl is not None:
        token = token.with_ttl(ttl)
    return token.to_jwt()
