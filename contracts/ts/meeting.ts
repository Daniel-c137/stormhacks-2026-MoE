export type MeetingStatus = "scheduled" | "live" | "processing" | "needs_review" | "pushed";
export type ParticipantRole = "host" | "member";

/** A team member account. */
export interface Person {
  id: string;
  name: string;
  short: string;
  initials: string;
  title?: string | null;
  email?: string | null;
  photo_url?: string | null;
}

export interface Team {
  id: string;
  name: string;
  member_ids: string[];
  github_repo?: string | null;
  jira_project?: string | null;
}

/** A scheduled meeting has scheduled_start and no started_at until someone starts it. */
export interface Meeting {
  id: string;
  team_id: string;
  title: string;
  status: MeetingStatus;
  code: string;
  host_id: string;
  participant_ids: string[];
  invitee_ids: string[];
  scheduled_start?: string | null; // ISO 8601
  started_at?: string | null; // ISO 8601
  ended_at?: string | null; // ISO 8601
  duration_min?: number | null;
  jira_keys: string[];
  transcript_deleted_at?: string | null; // ISO 8601; set once retention removed the segments
  agent_joined_at?: string | null; // ISO 8601; when the agent first joined; null if it never did
  translate: boolean; // live translation of non-English speech (#106); fixed once someone joins
}

/** Live participant view; id is the account id and the LiveKit identity. */
export interface Participant {
  id: string;
  name: string;
  role: ParticipantRole;
  is_agent: boolean;
  mic_on: boolean;
  cam_on: boolean;
  is_speaking: boolean;
  hand_raised: boolean;
}
