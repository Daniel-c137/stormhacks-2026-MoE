from typing import Protocol

from contracts import Agenda, AgendaNudge, Meeting, TranscriptSegment


class AgendaPlanner(Protocol):
    async def build(self, meeting: Meeting) -> Agenda:
        """From previous summaries, open tasks and unfinished GitHub/Jira work."""
        ...

    async def track(
        self, agenda: Agenda, segments: list[TranscriptSegment], minutes_left: float
    ) -> tuple[Agenda, list[AgendaNudge]]:
        """Mark covered items; nudge about items that have not come up near the end."""
        ...
