"use client";

import {
  type CodeRepo,
  type CodeRepoChoice,
  type ConnectorName,
  type ConnectorState,
  type ConnectorsUpdate,
  MAX_CODE_REPOS,
  type TeamSettings,
  identity,
} from "@moe/contracts";
import { type FormEvent, useCallback, useRef, useState } from "react";
import { Icon, type IconName } from "@/components/ui/Icon";
import { useDismiss } from "@/hooks/useDismiss";
import type { useConnectors } from "@/hooks/useApi";
import {
  connectGitHubRepo,
  connectJiraAccount,
  describeError,
  disconnectJiraAccount,
  updateConnectors,
} from "@/lib/api";

type Statuses = ReturnType<typeof useConnectors>;

const STATE: Record<ConnectorState, string> = {
  connected: "Connected",
  not_configured: "Not set up",
  failing: "Not reachable",
};

const ADD: { name: ConnectorName; label: string }[] = [
  { name: "github", label: "GitHub repository" },
  { name: "gitlab", label: "GitLab project" },
  { name: "jira", label: "Jira project" },
];

/** A GitLab project being typed: a new one (index null) or an edit of one already connected. */
interface Draft {
  index: number | null;
  path: string;
  ref: string;
}

/** A GitHub repository being connected with its token; `editing` is the path of one already
 * connected, whose branch or token changes (a blank token keeps its own). */
interface GitHubDraft {
  editing: string | null;
  repo: string;
  ref: string;
  token: string;
}

/** The Jira project being connected: where, which project, and the account it is read and
 * pushed with. */
interface JiraDraft {
  site: string;
  project: string;
  email: string;
  token: string;
}

const TOKENS_URL = "https://id.atlassian.com/manage-profile/security/api-tokens";
/** Where GitHub makes a fine-grained personal access token. */
const GITHUB_TOKENS_URL = "https://github.com/settings/personal-access-tokens/new";
/** The account row's key in the open-menu state, beside the `name:index` keys of the others. */
const ACCOUNT = "jira-account";

const choices = (repos: CodeRepo[]): CodeRepoChoice[] => repos.map(({ path, ref }) => ({ path, ref }));

/** The choice as saved, for PUT /settings/connectors (which replaces it whole). */
function current(s: TeamSettings): ConnectorsUpdate {
  return {
    github: choices(s.github.repos),
    gitlab: choices(s.gitlab.projects),
    jira: { site: s.jira.site ?? null, project: s.jira.project ?? null },
  };
}

/** Whether the project the agent reads is the one the Jira account was connected with. */
function readWithAccount(s: TeamSettings): boolean {
  const { project, site, connected, account_project, account_site } = s.jira;
  return (
    connected &&
    Boolean(project) &&
    project?.toUpperCase() === account_project?.toUpperCase() &&
    (!site || site.toLowerCase() === account_site?.toLowerCase())
  );
}

function State({ name, repo, statuses }: { name: ConnectorName; repo?: string; statuses: Statuses }) {
  const status = statuses.data?.find(
    (c) => c.name === name && (!repo || !c.repo || c.repo.toLowerCase() === repo.toLowerCase()),
  );
  const state = status?.state ?? (statuses.error ? "failing" : null);
  return (
    <span className="conn-state" data-state={state ?? "checking"} title={status?.detail ?? (statuses.error ? describeError(statuses.error) : undefined)}>
      {state ? STATE[state] : "Checking…"}
    </span>
  );
}

/** One row per connected repository, project and Jira project; admins add, edit and remove them
 * inline. Each change is saved at once and the states are checked again. A GitHub repository is
 * connected with the fine-grained token it is read with (PUT /settings/github/repos), each its
 * own; a Jira project with the account it is read and pushed with (PUT /settings/jira/account). */
export function Connectors({
  settings,
  statuses,
  canEdit,
  onSaved,
}: {
  settings: TeamSettings;
  statuses: Statuses;
  canEdit: boolean;
  onSaved: (saved: TeamSettings) => void;
}) {
  const [draft, setDraft] = useState<Draft | null>(null);
  const [github, setGithub] = useState<GitHubDraft | null>(null);
  const [jira, setJira] = useState<JiraDraft | null>(null);
  const [adding, setAdding] = useState(false);
  const [menu, setMenu] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [problem, setProblem] = useState("");
  const addRef = useRef<HTMLDivElement>(null);
  const rowMenu = useRef<HTMLDivElement>(null);
  useDismiss(addRef, adding, useCallback(() => setAdding(false), []));
  useDismiss(rowMenu, menu !== null, useCallback(() => setMenu(null), []));

  /** Runs a change and shows what it saved; `failed` prefixes the reason it did not. */
  const run = async (change: () => Promise<TeamSettings>, failed: string, done?: () => void) => {
    setMenu(null);
    setBusy(true);
    setProblem("");
    try {
      onSaved(await change());
      done?.();
      statuses.reload();
    } catch (err) {
      setProblem(`${failed} ${describeError(err)}`);
    } finally {
      setBusy(false);
    }
  };

  const save = (next: ConnectorsUpdate) => run(() => updateConnectors(next), "Not saved.", () => setDraft(null));

  const close = () => {
    setAdding(false);
    setMenu(null);
    setDraft(null);
    setGithub(null);
    setJira(null);
    setProblem("");
  };

  const startGitHub = (repo: CodeRepo | null = null) => {
    close();
    setGithub({ editing: repo?.path ?? null, repo: repo?.path ?? "", ref: repo?.ref ?? "", token: "" });
  };

  const startGitLab = (index: number | null = null) => {
    close();
    const project = index === null ? null : settings.gitlab.projects[index];
    setDraft({ index, path: project?.path ?? "", ref: project?.ref ?? "" });
  };

  const startJira = (fromAccount = false) => {
    close();
    const j = settings.jira;
    setJira({
      site: (fromAccount ? j.account_site : j.site) ?? j.account_site ?? "",
      project: (fromAccount ? j.account_project : j.project) ?? "",
      email: j.account_email ?? "",
      token: "",
    });
  };

  const start = (name: ConnectorName) => {
    if (name === "github") startGitHub();
    else if (name === "gitlab") startGitLab();
    else startJira();
  };

  const connectGitHub = (e: FormEvent) => {
    e.preventDefault();
    if (!github) return;
    void run(
      () => connectGitHubRepo({ repo: github.repo.trim(), ref: github.ref.trim() || null, token: github.token.trim() }),
      "Not connected.",
      () => setGithub(null),
    );
  };

  const connectJira = (e: FormEvent) => {
    e.preventDefault();
    if (!jira) return;
    void run(
      () =>
        connectJiraAccount({
          site: jira.site.trim(),
          project: jira.project.trim().toUpperCase(),
          email: jira.email.trim(),
          api_token: jira.token.trim(),
        }),
      "Not connected.",
      () => setJira(null),
    );
  };

  const remove = (name: ConnectorName, index: number) => {
    const next = current(settings);
    if (name === "jira") {
      if (readWithAccount(settings)) {
        void run(disconnectJiraAccount, "Not removed.");
        return;
      }
      next.jira = { site: null, project: null };
    } else next[name] = next[name].filter((_, i) => i !== index);
    setMenu(null);
    void save(next);
  };

  const submitGitLab = (e: FormEvent) => {
    e.preventDefault();
    if (!draft) return;
    const next = current(settings);
    const path = draft.path.trim();
    const ref = draft.ref.trim() || null;
    next.gitlab =
      draft.index === null ? [...next.gitlab, { path, ref }] : next.gitlab.map((r, i) => (i === draft.index ? { path, ref } : r));
    void save(next);
  };

  const pushedTo = readWithAccount(settings) ? ` · tasks are pushed here as ${settings.jira.account_email}` : "";
  const rows: { name: ConnectorName; index: number; label: string; detail: string | null }[] = [
    ...settings.github.repos.map((r, index) => ({
      name: "github" as const,
      index,
      label: r.path,
      detail: [r.ref, r.login && `@${r.login}`].filter(Boolean).join(" · ") || null,
    })),
    ...settings.gitlab.projects.map((r, index) => ({ name: "gitlab" as const, index, label: r.path, detail: r.ref ?? null })),
    ...(settings.jira.project
      ? [{ name: "jira" as const, index: 0, label: settings.jira.project, detail: `${settings.jira.site ?? ""}${pushedTo}` || null }]
      : []),
  ];
  /** A Jira account connected for another project than the one read (before the two were one). */
  const separateAccount = settings.jira.connected && !readWithAccount(settings);
  const full: Record<ConnectorName, boolean> = {
    github: settings.github.repos.length >= MAX_CODE_REPOS,
    gitlab: settings.gitlab.projects.length >= MAX_CODE_REPOS,
    jira: Boolean(settings.jira.project) || settings.jira.connected,
  };

  const gitlabForm = (d: Draft) => (
    <form className="conn-row conn-edit" onSubmit={submitGitLab}>
      <Icon name="gitlab" className="conn-icon is-gitlab" />
      <input
        className="field mono"
        aria-label="GitLab project"
        value={d.path}
        onChange={(e) => setDraft({ ...d, path: e.target.value })}
        placeholder="group/project"
        autoFocus
        required
      />
      <input
        className="field mono conn-ref"
        aria-label="Branch or tag"
        value={d.ref}
        onChange={(e) => setDraft({ ...d, ref: e.target.value })}
        placeholder="branch"
      />
      <button type="submit" className="btn btn-primary btn-sm" disabled={busy || !d.path.trim()}>
        {busy ? <Icon name="loader-circle" className="spin" /> : <Icon name="check" />}
        Save
      </button>
      <button type="button" className="icon-btn sm" onClick={() => setDraft(null)} aria-label="Cancel">
        <Icon name="x" />
      </button>
    </form>
  );

  const githubForm = (d: GitHubDraft) => (
    <form className="conn-account" onSubmit={connectGitHub} aria-label="Connect a GitHub repository">
      <h3>{d.editing ? `Change ${d.editing}` : "Connect a GitHub repository"}</h3>
      <label className="label">
        Repository
        <input
          className="field mono"
          value={d.repo}
          onChange={(e) => setGithub({ ...d, repo: e.target.value })}
          placeholder="owner/repo"
          readOnly={d.editing !== null}
          autoFocus={d.editing === null}
          required
        />
      </label>
      <label className="label">
        Branch or tag
        <input
          className="field mono"
          value={d.ref}
          onChange={(e) => setGithub({ ...d, ref: e.target.value })}
          placeholder="default branch"
        />
      </label>
      <label className="label wide">
        Fine-grained access token
        <input
          className="field mono"
          type="password"
          value={d.token}
          onChange={(e) => setGithub({ ...d, token: e.target.value })}
          placeholder={d.editing ? "Leave blank to keep the current token" : "github_pat_…"}
          autoComplete="new-password"
          autoFocus={d.editing !== null}
          required={d.editing === null}
        />
      </label>
      <p className="note">
        <a className="link" href={GITHUB_TOKENS_URL} target="_blank" rel="noopener noreferrer">
          Create a token on GitHub
        </a>{" "}
        with read-only access to this repository: Metadata, Contents, Issues, Pull requests and Commit statuses. It is
        checked with GitHub, stored encrypted and never shown again.
      </p>
      <div className="conn-account-actions">
        <button
          type="submit"
          className="btn btn-primary btn-sm"
          disabled={busy || !d.repo.trim() || (d.editing === null && !d.token.trim())}
        >
          {busy ? <Icon name="loader-circle" className="spin" /> : <Icon name="link" />}
          {d.editing ? "Save" : "Connect"}
        </button>
        <button type="button" className="btn btn-quiet btn-sm" onClick={() => setGithub(null)} disabled={busy}>
          Cancel
        </button>
      </div>
    </form>
  );

  const jiraForm = (d: JiraDraft) => (
    <form className="conn-account" onSubmit={connectJira} aria-label="Connect a Jira project">
      <h3>{settings.jira.project || settings.jira.connected ? "Change the Jira project" : "Connect a Jira project"}</h3>
      <label className="label">
        Jira site
        <input
          className="field mono"
          value={d.site}
          onChange={(e) => setJira({ ...d, site: e.target.value })}
          placeholder="your-team.atlassian.net"
          autoFocus
          required
        />
      </label>
      <label className="label">
        Project key
        <input
          className="field mono"
          value={d.project}
          onChange={(e) => setJira({ ...d, project: e.target.value })}
          placeholder="KEY"
          style={{ textTransform: "uppercase" }}
          required
        />
      </label>
      <label className="label">
        Atlassian account email
        <input
          className="field"
          type="email"
          value={d.email}
          onChange={(e) => setJira({ ...d, email: e.target.value })}
          placeholder="you@company.com"
          autoComplete="off"
          required
        />
      </label>
      <label className="label">
        API token
        <input
          className="field mono"
          type="password"
          value={d.token}
          onChange={(e) => setJira({ ...d, token: e.target.value })}
          autoComplete="new-password"
          required
        />
      </label>
      <p className="note">
        {identity.agent_name} reads this project&apos;s issues, and approved tasks are created in it, as this account.
        Create a token at{" "}
        <a className="link" href={TOKENS_URL} target="_blank" rel="noopener noreferrer">
          id.atlassian.com
        </a>{" "}
        (Security, API tokens). It is checked with Jira, stored encrypted and never shown again.
      </p>
      <div className="conn-account-actions">
        <button
          type="submit"
          className="btn btn-primary btn-sm"
          disabled={busy || !d.site.trim() || !d.project.trim() || !d.email.trim() || !d.token.trim()}
        >
          {busy ? <Icon name="loader-circle" className="spin" /> : <Icon name="link" />}
          Connect
        </button>
        <button type="button" className="btn btn-quiet btn-sm" onClick={() => setJira(null)} disabled={busy}>
          Cancel
        </button>
      </div>
    </form>
  );

  const rowMenuButton = (key: string, label: string, items: { label: string; icon: IconName; danger?: boolean; act: () => void }[]) => (
    <div className="conn-more" ref={menu === key ? rowMenu : undefined}>
      <button
        type="button"
        className="icon-btn sm"
        aria-label={`More for ${label}`}
        aria-haspopup="menu"
        aria-expanded={menu === key}
        onClick={() => setMenu(menu === key ? null : key)}
        disabled={busy}
      >
        <Icon name="ellipsis" />
      </button>
      {menu === key && (
        <div className="menu conn-menu" role="menu" aria-label={label}>
          {items.map((item) => (
            <button
              key={item.label}
              type="button"
              className={item.danger ? "menu-item danger" : "menu-item"}
              role="menuitem"
              onClick={item.act}
            >
              <Icon name={item.icon} />
              {item.label}
            </button>
          ))}
        </div>
      )}
    </div>
  );

  return (
    <div className="conn">
      <ul className="conn-list">
        {rows.map((row) => {
          const key = `${row.name}:${row.index}`;
          if (row.name === "github" && github?.editing === row.label) return <li key={key}>{githubForm(github)}</li>;
          if (row.name === "gitlab" && draft?.index === row.index) return <li key={key}>{gitlabForm(draft)}</li>;
          if (row.name === "jira" && jira && !separateAccount) return <li key={key}>{jiraForm(jira)}</li>;
          const edit = () =>
            row.name === "github"
              ? startGitHub(settings.github.repos[row.index])
              : row.name === "gitlab"
                ? startGitLab(row.index)
                : startJira();
          return (
            <li key={key} className="conn-row">
              <Icon name={row.name} className={`conn-icon is-${row.name}`} />
              <span className="conn-name">
                {row.label}
                {row.detail && <span className="conn-ref-text"> · {row.detail}</span>}
              </span>
              <State name={row.name} repo={row.name === "github" ? row.label : undefined} statuses={statuses} />
              {canEdit &&
                rowMenuButton(key, row.label, [
                  { label: "Edit", icon: "pencil", act: edit },
                  { label: "Remove", icon: "trash-2", danger: true, act: () => remove(row.name, row.index) },
                ])}
            </li>
          );
        })}
        {github && github.editing === null && <li>{githubForm(github)}</li>}
        {draft && draft.index === null && <li>{gitlabForm(draft)}</li>}
        {jira && !settings.jira.project && !separateAccount && <li>{jiraForm(jira)}</li>}
        {separateAccount && !jira && (
          <li className="conn-row">
            <Icon name="jira" className="conn-icon is-jira" />
            <span className="conn-name">
              {settings.jira.account_project}
              <span className="conn-ref-text">
                {" "}
                · {settings.jira.account_site} · tasks are pushed here as {settings.jira.account_email}
              </span>
            </span>
            <span className="conn-state" data-state="connected">
              Connected
            </span>
            {canEdit &&
              rowMenuButton(ACCOUNT, "Jira account", [
                { label: "Change", icon: "pencil", act: () => startJira(true) },
                { label: "Disconnect", icon: "trash-2", danger: true, act: () => void run(disconnectJiraAccount, "Not disconnected.") },
              ])}
          </li>
        )}
        {separateAccount && jira && <li>{jiraForm(jira)}</li>}
      </ul>
      {canEdit && (
        <div className="conn-add" ref={addRef}>
          <button
            type="button"
            className="btn btn-outline btn-sm"
            aria-haspopup="menu"
            aria-expanded={adding}
            onClick={() => setAdding((open) => !open)}
            disabled={busy}
          >
            <Icon name="plus" />
            Add connector
            <Icon name="chevron-down" />
          </button>
          {adding && (
            <div className="menu" role="menu" aria-label="Add connector">
              {ADD.map(({ name, label }) => (
                <button key={name} type="button" className="menu-item" role="menuitem" disabled={full[name]} onClick={() => start(name)}>
                  <Icon name={name} className={`conn-icon is-${name}`} />
                  {label}
                </button>
              ))}
            </div>
          )}
        </div>
      )}
      {problem && (
        <p className="err" role="alert">
          {problem}
        </p>
      )}
    </div>
  );
}
