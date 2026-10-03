// LiveKit data-channel topics and the payload each one carries.
// Public chat uses LiveKit's built-in chat topic. Private questions and answers go over HTTP to
// the brain, never through a room broadcast.
import type { Agenda, AgendaNudge } from "./agenda";
import type { AgentState, FactCheck, ResponseAction, ResponseCard } from "./agent";
import type { TranscriptSegment } from "./transcript";

export const Topic = {
  TRANSCRIPT: "transcript", // realtime -> room
  AGENT_STATE: "agent.state", // realtime -> room
  RESPONSE_CARD: "agent.card", // realtime -> room
  RESPONSE_ACTION: "agent.card.action", // board -> realtime
  FACT_CHECK: "agent.fact_check", // realtime -> room, or one participant when private
  AGENDA: "agent.agenda", // realtime -> room
  AGENDA_NUDGE: "agent.agenda.nudge", // realtime -> room
  STAGE: "stage", // board -> room: snippet id shown on stage, or null
} as const;
export type Topic = (typeof Topic)[keyof typeof Topic];

export interface StagePayload {
  snippet_id: string | null;
}

export interface TopicPayloads {
  [Topic.TRANSCRIPT]: TranscriptSegment;
  [Topic.AGENT_STATE]: AgentState;
  [Topic.RESPONSE_CARD]: ResponseCard;
  [Topic.RESPONSE_ACTION]: ResponseAction;
  [Topic.FACT_CHECK]: FactCheck;
  [Topic.AGENDA]: Agenda;
  [Topic.AGENDA_NUDGE]: AgendaNudge;
  [Topic.STAGE]: StagePayload;
}
