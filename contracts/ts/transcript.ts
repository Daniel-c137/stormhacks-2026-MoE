/**
 * One speaker-attributed caption. Partial segments are transient; only final ones are saved.
 * t_start and t_end are seconds from the meeting start.
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
}
