"""Supabase Auth access-token verification."""


class InvalidToken(Exception):
    """The bearer token is not a valid access token for this project."""


class Jwks:
    """The project's public signing keys."""

    def __init__(self, url, fetch=None, ttl=600.0, clock=None):
        raise NotImplementedError

    async def key(self, kid: str):
        raise NotImplementedError
