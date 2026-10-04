export type Sensitivity = "quiet" | "balanced" | "eager";
export type WhoCanAllow = "everyone" | "host";
export type ConnectorName = "github" | "gitlab" | "jira";
export type ConnectorState = "connected" | "not_configured" | "failing";

/** Most repositories (GitHub) or projects (GitLab) one team connects, each. */
export const MAX_CODE_REPOS = 10;

/** One connected repository: a GitHub owner/name or a GitLab project path (group/project,
 * subgroups included), read at `ref` (null: the default branch). Connection and index state
 * belong to the server. */
export interface CodeRepo {
  path: string;
  ref?: string | null;
  connected: boolean;
  indexed_at?: string | null;
  files?: number | null;
}

export interface GitHubSettings {
  repos: CodeRepo[]; // owner/name each
}

export interface GitLabSettings {
  projects: CodeRepo[]; // group/project each
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
  default_label?: string | null; // only on the agent's default voice, naming the agent
}

export interface TeamSettings {
  team_id: string;
  // Connectors change only at PUT /settings/connectors; PUT /settings keeps them as saved.
  github: GitHubSettings;
  gitlab: GitLabSettings;
  jira: JiraSettings;
  voice?: string | null;
  wake_phrase?: string | null; // null means the identity's default wake phrase
  sensitivity: Sensitivity;
  interrupt_minutes: number;
  who_can_allow: WhoCanAllow;
  timezone: string; // IANA name, e.g. "America/Vancouver"; dates people see use it
}

/** A repository or project an admin connects: its path and, optionally, a branch or tag. */
export interface CodeRepoChoice {
  path: string;
  ref?: string | null;
}

export interface JiraChoice {
  site?: string | null;
  project?: string | null;
}

/** PUT /settings/connectors (admins only): the whole choice, replacing the saved one.
 * Repositories already connected keep their connection and index state. */
export interface ConnectorsUpdate {
  github: CodeRepoChoice[];
  gitlab: CodeRepoChoice[];
  jira: JiraChoice;
}

/** Whether an integration can be used right now. Failing and unconfigured are never hidden. */
export interface ConnectorStatus {
  name: ConnectorName;
  state: ConnectorState;
  detail?: string | null;
}
