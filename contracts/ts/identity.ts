// Working product and agent names. Rename them in contracts/identity.json only.
import raw from "../identity.json";

export interface Identity {
  product_name: string;
  agent_name: string;
}

export const identity: Identity = raw;

// Stable LiveKit identity for the agent. Never derived from the display name.
export const AGENT_PARTICIPANT_ID = "agent";

// The agent's name alone: "Polaris, what's blocking DS-104?".
export const wakePhrase = (id: Identity = identity) => id.agent_name;
export const mention = (id: Identity = identity) => `@${id.agent_name}`;
