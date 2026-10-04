import re
from datetime import datetime

from pydantic import BaseModel

from contracts import AGENT_PARTICIPANT_ID, Person, Report, TranscriptSegment


class TranscriptInput(BaseModel):
    """A meeting transcript to process: its segments and, when known, who was there."""

    meeting_id: str
    title: str
    started_at: datetime | None = None
    members: list[Person] = []
    segments: list[TranscriptSegment]
    agent_joined: bool = False  # the worker recorded the agent joining the meeting

    def agent_attended(self) -> bool:
        """The agent was there if it was recorded joining or it spoke."""
        return self.agent_joined or any(s.speaker_id == AGENT_PARTICIPANT_ID for s in self.segments)

    def final_segments(self) -> list[TranscriptSegment]:
        """Final segments in time order, once each. Partial captions never reach the report."""
        seen: set[str] = set()
        final = []
        for segment in sorted(self.segments, key=lambda s: s.t_start):
            if segment.is_final and segment.seg_id not in seen:
                seen.add(segment.seg_id)
                final.append(segment)
        return final

    def people(self) -> list[Person]:
        """The given members, or else everyone who spoke. Never the agent."""
        if self.members:
            return [p for p in self.members if p.id != AGENT_PARTICIPANT_ID]
        people: dict[str, Person] = {}
        for segment in self.segments:
            if segment.speaker_id != AGENT_PARTICIPANT_ID and segment.speaker_id not in people:
                people[segment.speaker_id] = person_from_name(
                    segment.speaker_id, segment.speaker_name
                )
        return list(people.values())


class ProcessedMeeting(BaseModel):
    """The review file: `brain report` writes it, a person edits it, `brain push` reads it."""

    meeting_id: str
    title: str
    started_at: datetime | None = None
    timezone: str = "UTC"  # the team's IANA time zone; the meeting is dated in it
    members: list[Person]
    report: Report


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.casefold()).strip("-")


def person_from_name(person_id: str, name: str) -> Person:
    words = name.split()
    return Person(
        id=person_id,
        name=name,
        short=words[0] if words else name,
        initials="".join(w[0] for w in words[:2]).upper(),
    )
