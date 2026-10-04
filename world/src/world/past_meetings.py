"""The demo world's past meetings: mock-data/meetings/*.md, YAML front matter then transcript lines
`[HH:MM:SS] Speaker: text`, the time an offset from the meeting's start."""

import re
from datetime import date
from pathlib import Path

import yaml
from pydantic import AwareDatetime, BaseModel, ValidationError

from brain.report.models import slug
from contracts import TranscriptSegment

from .config import MOCK_DATA_DIR, SnapshotName, world_spec
from .model import PastMeeting

SEED_MEETINGS_DIR = MOCK_DATA_DIR / "meetings"
LAST_SEGMENT_SECONDS = 5.0
"""How long the last line is taken to last; every other line lasts until the next one starts."""

FRONT_MATTER = re.compile(r"\A---\n(?P<yaml>.*?)\n---\n", re.DOTALL)
LINE = re.compile(
    r"^\[(?P<h>\d{2}):(?P<m>[0-5]\d):(?P<s>[0-5]\d)\] (?P<speaker>[^:]+?): (?P<text>\S.*)$"
)
SKIPPED = re.compile(r"^(#.*|\[[^\]]*\]|Attendees: .*)$")
"""Markdown headings, bracketed notes like [Recording paused], and the attendees line."""


class MeetingFileError(ValueError):
    """A meeting file that cannot be read; the message names the file and line."""


class SeedParticipant(BaseModel):
    name: str
    role: str


class FrontMatter(BaseModel):
    meeting_id: str
    title: str
    date: date
    start: AwareDatetime
    end: AwareDatetime
    timezone: str
    participants: list[SeedParticipant]


class SeedMeeting(PastMeeting):
    """A past meeting as its file gives it. `id` is the file's meeting_id, not a store id."""

    day: date
    ended_at: AwareDatetime
    timezone: str
    participants: list[SeedParticipant]


def person_id(name: str) -> str:
    """The stable id of a world person, e.g. "Mohammad Reza" -> "p-mohammad-reza"."""
    return f"p-{slug(name)}"


def load_meeting(path: Path) -> SeedMeeting:
    text = path.read_text()
    match = FRONT_MATTER.match(text)
    if match is None:
        raise MeetingFileError(f"{path}: no front matter between --- lines")
    try:
        # BaseLoader keeps every value a string; pydantic parses dates and offsets.
        front = FrontMatter.model_validate(yaml.load(match["yaml"], Loader=yaml.BaseLoader))
    except (yaml.YAMLError, ValidationError) as e:
        raise MeetingFileError(f"{path}: bad front matter: {e}") from None
    if front.end <= front.start:
        raise MeetingFileError(f"{path}: the meeting ends before it starts")

    speakers = {p.name: person_id(p.name) for p in front.participants}
    first_line = text[: match.end()].count("\n") + 1
    turns: list[tuple[float, str, str]] = []
    for number, raw in enumerate(text[match.end() :].splitlines(), start=first_line):
        line = raw.strip()
        if not line or SKIPPED.match(line):
            continue
        parsed = LINE.match(line)
        if parsed is None:
            raise MeetingFileError(
                f"{path}:{number}: expected '[HH:MM:SS] Speaker: text', got {line[:60]!r}"
            )
        speaker = parsed["speaker"]
        if speaker not in speakers:
            raise MeetingFileError(
                f"{path}:{number}: {speaker!r} is not among the participants"
                f" ({', '.join(speakers)})"
            )
        t = float(int(parsed["h"]) * 3600 + int(parsed["m"]) * 60 + int(parsed["s"]))
        if turns and t < turns[-1][0]:
            raise MeetingFileError(
                f"{path}:{number}: the time goes back to {parsed['h']}:{parsed['m']}:{parsed['s']}"
            )
        turns.append((t, speaker, parsed["text"]))
    if not turns:
        raise MeetingFileError(f"{path}: no transcript lines")

    ends = [t for t, _, _ in turns[1:]] + [turns[-1][0] + LAST_SEGMENT_SECONDS]
    segments = [
        TranscriptSegment(
            seg_id=f"{front.meeting_id}-{n:04d}",
            meeting_id=front.meeting_id,
            speaker_id=speakers[speaker],
            speaker_name=speaker,
            text=line,
            is_final=True,
            t_start=t,
            t_end=end,
        )
        for n, ((t, speaker, line), end) in enumerate(zip(turns, ends, strict=True), start=1)
    ]
    return SeedMeeting(
        id=front.meeting_id,
        title=front.title,
        started_at=front.start,
        ended_at=front.end,
        day=front.date,
        timezone=front.timezone,
        participants=front.participants,
        participant_ids=list(speakers.values()),
        segments=segments,
    )


def load_meetings(directory: Path = SEED_MEETINGS_DIR) -> list[SeedMeeting]:
    """Every meeting file in the directory, oldest first."""
    meetings = [load_meeting(path) for path in sorted(directory.glob("*.md"))]
    return sorted(meetings, key=lambda m: m.started_at)


def snapshot_meetings(name: SnapshotName, directory: Path = SEED_MEETINGS_DIR) -> list[SeedMeeting]:
    """The meetings a snapshot has happened by: dated on or before its data_until, oldest first."""
    until = world_spec().snapshots[name].data_until
    return [m for m in load_meetings(directory) if m.day <= until]
