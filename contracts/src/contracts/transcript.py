from pydantic import BaseModel


class TranscriptSegment(BaseModel):
    """One speaker-attributed caption. Partial segments are transient; only final ones are saved.

    t_start and t_end are seconds from the meeting start. `text` is always what everyone reads,
    in English. For speech in another language, `language` is its ISO 639-1 code and
    `original_text` the words as said, with `text` their English translation; a provisional
    translation is a partial. Non-English `language` with no `original_text` means the
    translation failed and `text` is the original, untranslated.
    """

    seg_id: str
    meeting_id: str
    speaker_id: str
    speaker_name: str
    text: str
    is_final: bool
    t_start: float
    t_end: float
    language: str | None = None
    original_text: str | None = None
