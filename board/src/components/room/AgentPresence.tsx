import type { AgentState } from "@moe/contracts";

export interface AgentPresenceProps {
  state: AgentState;
}

/** Agent status: listening, capturing, working, hand raised (normal or critical), speaking, follow-up window. */
export function AgentPresence(_props: AgentPresenceProps) {
  return null;
}
