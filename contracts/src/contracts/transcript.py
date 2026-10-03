from pydantic import BaseModel


class TranscriptSegment(BaseModel):
    """One speaker-attributed caption. Partial segments are transient; only final ones are saved.

    t_start and t_end are seconds from the meeting start.
    """

    seg_id: str
    meeting_id: str
    speaker_id: str
    speaker_name: str
    text: str
    is_final: bool
    t_start: float
    t_end: float
