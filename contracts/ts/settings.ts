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
  /** The GitHub account whose token an admin connected (PUT /settings/github/account), by its
   * login: the team's reads then go to GitHub's hosted MCP server with that token. null: the
   * server's GITHUB_MCP_URL, with no credentials. The token never leaves the brain. */
  account_login?: string | null;
}

export interface GitLabSettings {
  projects: CodeRepo[]; // group/project each
}

export interface JiraSettings {
  /** The project the agent reads (answers, agenda suggestions, fact checks) and its site. */
  site?: string | null;
  project?: string | null;
  /** The account an admin connected for pushing (PUT /settings/jira/account): approved task
   * drafts become issues in `account_project` on `account_site`, as `account_email`. Its own
   * site and project, so connecting it changes nothing the agent reads. Its API token never
   * leaves the brain. */
  connected: boolean;
  account_email?: string | null;
  account_site?: string | null;
  account_project?: string | null;
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

/** PUT /settings/jira/account (admins only): the Jira Cloud site (name.atlassian.net), the
 * Atlassian account's email and API token, and the project issues are created in. The brain
 * checks them against Jira before saving; the token is stored encrypted and never returned. */
export interface JiraAccountConnect {
  site: string;
  email: string;
  api_token: string;
  project: string;
}

/** PUT /settings/github/account (admins only): a fine-grained personal access token with read
 * access to the team's repositories. The brain checks it with GitHub before saving; it is stored
 * encrypted and never returned. */
export interface GitHubAccountConnect {
  token: string;
}

/** PUT /settings/connectors (admins only): the whole choice, replacing the saved one.
 * Repositories already connected keep their connection and index state. The Jira account
 * connected for pushing is separate and stays as it is. */
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
