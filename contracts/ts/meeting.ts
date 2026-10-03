export type MeetingStatus = "live" | "processing" | "needs_review" | "pushed";
export type ParticipantRole = "host" | "member";

/** A team member account. */
export interface Person {
  id: string;
  name: string;
  short: string;
  initials: string;
  title?: string | null;
}

export interface Team {
  id: string;
  name: string;
  member_ids: string[];
  github_repo?: string | null;
  jira_project?: string | null;
}

export interface Meeting {
  id: string;
  team_id: string;
  title: string;
  status: MeetingStatus;
  code: string;
  host_id: string;
  participant_ids: string[];
  started_at?: string | null; // ISO 8601
  duration_min?: number | null;
  jira_keys: string[];
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
