from typing import Protocol

from contracts import FactCheck, Meeting, TranscriptSegment


class FactChecker(Protocol):
    """Checks one technical claim against code, GitHub, Jira and past meetings."""

    async def check(self, meeting: Meeting, segment: TranscriptSegment) -> FactCheck | None: ...
