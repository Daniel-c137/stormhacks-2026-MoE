export type Sensitivity = "quiet" | "balanced" | "eager";
export type WhoCanAllow = "everyone" | "host";
export type ConnectorName = "github" | "jira";
export type ConnectorState = "connected" | "not_configured" | "failing";

export interface GitHubSettings {
  repo?: string | null;
  ref?: string | null;
  connected: boolean;
  indexed_at?: string | null;
  files?: number | null;
}

export interface JiraSettings {
  site?: string | null;
  project?: string | null;
  connected: boolean;
}

export interface Voice {
  id: string;
  name: string;
  desc: string;
  sample: string;
}

export interface TeamSettings {
  team_id: string;
  github: GitHubSettings;
  jira: JiraSettings;
  voice?: string | null;
  wake_phrase?: string | null; // null means the identity's default wake phrase
  sensitivity: Sensitivity;
  interrupt_minutes: number;
  who_can_allow: WhoCanAllow;
  timezone: string; // IANA name, e.g. "America/Vancouver"; dates people see use it
}

/** Whether an integration can be used right now. Failing and unconfigured are never hidden. */
export interface ConnectorStatus {
  name: ConnectorName;
  state: ConnectorState;
  detail?: string | null;
}
