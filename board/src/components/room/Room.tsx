import type { JoinMeetingResponse } from "@moe/contracts";

export interface RoomProps {
  join: JoinMeetingResponse;
}

/** LiveKit room: participant grid or code stage, agent presence, side panel, controls. */
export function Room(_props: RoomProps) {
  return null;
}
