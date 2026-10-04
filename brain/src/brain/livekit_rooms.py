from typing import Protocol

from livekit.api import DeleteRoomRequest, ListParticipantsRequest, LiveKitAPI, TwirpError
from livekit.protocol.models import ParticipantInfo

from .config import Settings


class Rooms(Protocol):
    async def close(self, room: str) -> None:
        """Disconnect everyone and close the room. Raises if LiveKit can't be reached."""
        ...

    async def identities(self, room: str) -> list[str]:
        """The identities connected to the room now; none if it doesn't exist. Raises if LiveKit
        can't be reached."""
        ...


class LiveKitRooms:
    def __init__(self, url: str, api_key: str, api_secret: str):
        self._url, self._key, self._secret = url, api_key, api_secret

    async def close(self, room: str) -> None:
        api = LiveKitAPI(self._url, self._key, self._secret)
        try:
            await api.room.delete_room(DeleteRoomRequest(room=room))
        finally:
            await api.aclose()

    async def identities(self, room: str) -> list[str]:
        api = LiveKitAPI(self._url, self._key, self._secret)
        try:
            found = await api.room.list_participants(ListParticipantsRequest(room=room))
        except TwirpError as err:
            if err.code == "not_found":  # nobody has joined yet, or the room closed
                return []
            raise
        finally:
            await api.aclose()
        return [
            p.identity for p in found.participants if p.state != ParticipantInfo.State.DISCONNECTED
        ]


class NoRooms:
    """LiveKit isn't configured, so no token was ever issued and there is no room to close."""

    async def close(self, room: str) -> None:
        return None

    async def identities(self, room: str) -> list[str]:
        return []


def rooms_from_settings(settings: Settings) -> Rooms:
    if settings.livekit_url and settings.livekit_api_key and settings.livekit_api_secret:
        return LiveKitRooms(
            settings.livekit_url, settings.livekit_api_key, settings.livekit_api_secret
        )
    return NoRooms()
