from contracts import Meeting, Person


def participant_token(meeting: Meeting, person: Person, *, is_host: bool) -> str:
    """LiveKit access token; identity is the account id so captions map to the right person."""
    raise NotImplementedError
