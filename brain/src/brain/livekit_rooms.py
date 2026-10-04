from typing import Protocol

from livekit.api import DeleteRoomRequest, LiveKitAPI

from .config import Settings


class Rooms(Protocol):
    async def close(self, room: str) -> None:
        """Disconnect everyone and close the room. Raises if LiveKit can't be reached."""
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


class NoRooms:
    """LiveKit isn't configured, so no token was ever issued and there is no room to close."""

    async def close(self, room: str) -> None:
        return None


def rooms_from_settings(settings: Settings) -> Rooms:
    if settings.livekit_url and settings.livekit_api_key and settings.livekit_api_secret:
        return LiveKitRooms(
            settings.livekit_url, settings.livekit_api_key, settings.livekit_api_secret
        )
    return NoRooms()
