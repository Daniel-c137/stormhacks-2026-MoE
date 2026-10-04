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
  /** GitHub only: the account whose token this repository was connected with (PUT
   * /settings/github/repos), by its login; it is read through GitHub's hosted MCP server with
   * that token. null: read from the server's GITHUB_MCP_URL with no credentials. The token never
   * leaves the brain. */
  login?: string | null;
}

export interface GitHubSettings {
  repos: CodeRepo[]; // owner/name each
}

export interface GitLabSettings {
  projects: CodeRepo[]; // group/project each
}

export interface JiraSettings {
  /** The project the agent reads (answers, agenda suggestions, fact checks) and its site. */
  site?: string | null;
  project?: string | null;
  /** The account an admin connected the project with (PUT /settings/jira/account): approved
   * task drafts become issues in `account_project` on `account_site`, as `account_email`, and
   * when that is the project above the agent reads it with the account too. Its API token never
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

/** PUT /settings/jira/account (admins only): the team's Jira project, which the agent reads and
 * approved tasks are created in, with the Jira Cloud site (name.atlassian.net) and the Atlassian
 * account's email and API token. The brain checks them against Jira before saving; the token is
 * stored encrypted and never returned. */
export interface JiraAccountConnect {
  site: string;
  email: string;
  api_token: string;
  project: string;
}

/** PUT /settings/github/repos (admins only): a repository to connect, or one already connected to
 * change (matched by path), with the branch or tag it is read at and the fine-grained personal
 * access token it is read with. The brain checks that the token reads the repository before
 * saving; it is stored encrypted and never returned. A blank token keeps the one a connected
 * repository has. */
export interface GitHubRepoConnect {
  repo: string;
  ref?: string | null;
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

/** Whether an integration can be used right now. Failing and unconfigured are never hidden.
 * GitHub has one per repository, each read with its own token: `repo` names it. */
export interface ConnectorStatus {
  name: ConnectorName;
  state: ConnectorState;
  detail?: string | null;
  repo?: string | null;
}
