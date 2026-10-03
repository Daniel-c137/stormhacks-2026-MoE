"""Load a transcript from JSON (TranscriptInput) or text lines like `[mm:ss] Name: text`."""

import re
from datetime import datetime
from pathlib import Path

from contracts import AGENT_PARTICIPANT_ID, TranscriptSegment, get_identity

from .models import TranscriptInput, slug

LINE = re.compile(
    r"^(?:\[(?P<ts>\d{1,2}:\d{2}(?::\d{2})?)\]\s*)?(?P<speaker>[^:\[\]]{1,60}?):\s+(?P<text>.*\S)\s*$"
)


def load_transcript(path: Path, *, started_at: datetime | None = None) -> TranscriptInput:
    if path.suffix == ".json":
        meeting = TranscriptInput.model_validate_json(path.read_text())
        return meeting.model_copy(update={"started_at": started_at}) if started_at else meeting
    return parse_text_transcript(path.read_text(), meeting_id=path.stem, started_at=started_at)


def parse_text_transcript(
    text: str, *, meeting_id: str, started_at: datetime | None = None
) -> TranscriptInput:
    """Lines without a timestamp take the previous one; lines without a speaker continue the
    previous line. The first `# heading` is the title."""
    agent_name = get_identity().agent_name
    title: str | None = None
    turns: list[dict] = []
    t = 0.0
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            title = title or line.lstrip("#").strip()
            continue
        match = LINE.match(line)
        if match:
            if match["ts"]:
                t = seconds(match["ts"])
            speaker = match["speaker"].strip()
            is_agent = speaker.casefold() == agent_name.casefold()
            turns.append(
                {
                    "speaker_id": AGENT_PARTICIPANT_ID if is_agent else slug(speaker),
                    "speaker_name": agent_name if is_agent else speaker,
                    "text": match["text"],
                    "t_start": t,
                }
            )
        elif turns:
            turns[-1]["text"] += " " + line

    segments = [
        TranscriptSegment(
            seg_id=f"{meeting_id}-seg-{i}",
            meeting_id=meeting_id,
            is_final=True,
            t_end=turns[i]["t_start"] if i < len(turns) else turn["t_start"],
            **turn,
        )
        for i, turn in enumerate(turns, 1)
    ]
    return TranscriptInput(
        meeting_id=meeting_id, title=title or meeting_id, started_at=started_at, segments=segments
    )


def seconds(timestamp: str) -> float:
    total = 0
    for part in timestamp.split(":"):
        total = total * 60 + int(part)
    return float(total)
