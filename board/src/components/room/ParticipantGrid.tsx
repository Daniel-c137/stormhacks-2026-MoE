import type { Participant, TranscriptSegment } from "@moe/contracts";

export interface ParticipantGridProps {
  participants: Participant[];
  /** Latest caption per speaker_id, shown on that participant's tile. */
  captions: Record<string, TranscriptSegment>;
}

/** People only; the agent has no video tile. */
export function ParticipantGrid(_props: ParticipantGridProps) {
  return null;
}
