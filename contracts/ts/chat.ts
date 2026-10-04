import type { Visibility } from "./agent";

/**
 * Public messages are saved with the meeting. Private ones are ephemeral and never stored.
 *
 * A private message has visibility "private" and its recipient_id; a private question to the
 * agent uses the agent's id (AGENT_PARTICIPANT_ID) as recipient_id.
 */
export interface ChatMessage {
  id: string;
  meeting_id: string;
  sender_id: string;
  sender_name: string;
  is_agent: boolean;
  text: string;
  ts: string; // ISO 8601
  visibility: Visibility;
  recipient_id?: string | null;
  snippet_id?: string | null;
}
