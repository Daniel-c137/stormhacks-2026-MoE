"""Who spoke a segment, shared by the report, the agent and meeting memory."""

from contracts import AGENT_PARTICIPANT_ID, TranscriptSegment


def by_agent(segment: TranscriptSegment) -> bool:
    """Whether the agent spoke this segment. Its own words are never a claim or a source."""
    return segment.speaker_id == AGENT_PARTICIPANT_ID
