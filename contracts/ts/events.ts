// LiveKit data-channel topics and the payload each one carries.
// Public chat uses LiveKit's built-in chat topic. Private questions and answers with the agent go
// over HTTP to the brain, never through a room broadcast. A private message, between two people or
// a fact-check from the agent to whoever made the claim, is sent to that one participant only and
// is never stored.
import type { Agenda, AgendaNudge } from "./agenda";
import type { AgentState, ResponseAction, ResponseCard } from "./agent";
import type { ChatMessage } from "./chat";
import type { TranscriptSegment } from "./transcript";

export const Topic = {
  TRANSCRIPT: "transcript", // realtime -> room
  AGENT_STATE: "agent.state", // realtime -> room
  ASK: "agent.ask", // board -> realtime: the Ask button; the presser's next final segment is the question
  RESPONSE_CARD: "agent.card", // realtime -> room
  RESPONSE_ACTION: "agent.card.action", // board -> realtime
  AGENDA: "agent.agenda", // realtime -> room
  AGENDA_NUDGE: "agent.agenda.nudge", // realtime -> room
  STAGE: "stage", // board -> room: snippet id shown on stage, or null
  PRIVATE_CHAT: "chat.private", // board or realtime -> one participant; ephemeral
} as const;
export type Topic = (typeof Topic)[keyof typeof Topic];

export interface StagePayload {
  snippet_id: string | null;
}

export interface AskSignal {
  by_id: string;
  cancel: boolean; // true withdraws a pending Ask press
}

export interface TopicPayloads {
  [Topic.TRANSCRIPT]: TranscriptSegment;
  [Topic.AGENT_STATE]: AgentState;
  [Topic.ASK]: AskSignal;
  [Topic.RESPONSE_CARD]: ResponseCard;
  [Topic.RESPONSE_ACTION]: ResponseAction;
  [Topic.AGENDA]: Agenda;
  [Topic.AGENDA_NUDGE]: AgendaNudge;
  [Topic.STAGE]: StagePayload;
  [Topic.PRIVATE_CHAT]: ChatMessage;
}
