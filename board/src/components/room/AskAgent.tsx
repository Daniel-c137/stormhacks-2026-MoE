import type { AskRequest } from "@moe/contracts";

export interface AskAgentProps {
  onAsk: (request: AskRequest) => void;
}

/** Typed question to the agent, public or private. Labelled with identity.agent_name. */
export function AskAgent(_props: AskAgentProps) {
  return null;
}
