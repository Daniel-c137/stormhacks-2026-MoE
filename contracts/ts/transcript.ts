/**
 * One speaker-attributed caption. Partial segments are transient; only final ones are saved.
 * t_start and t_end are seconds from the meeting start. `text` is always what everyone reads,
 * in English. For speech in another language, `language` is its ISO 639-1 code and
 * `original_text` the words as said, with `text` their English translation; a provisional
 * translation is a partial. Non-English `language` with no `original_text` means the
 * translation failed and `text` is the original, untranslated.
 */
export interface TranscriptSegment {
  seg_id: string;
  meeting_id: string;
  speaker_id: string;
  speaker_name: string;
  text: string;
  is_final: boolean;
  t_start: number;
  t_end: number;
  language?: string | null;
  original_text?: string | null;
}
