"use client";

import {
  type CodeRepo,
  type CodeRepoChoice,
  type ConnectorName,
  type ConnectorState,
  type ConnectorsUpdate,
  MAX_CODE_REPOS,
  type TeamSettings,
} from "@moe/contracts";
import { type FormEvent, useCallback, useRef, useState } from "react";
import { Icon } from "@/components/ui/Icon";
import { useDismiss } from "@/hooks/useDismiss";
import type { useConnectors } from "@/hooks/useApi";
import { connectJiraAccount, describeError, disconnectJiraAccount, updateConnectors } from "@/lib/api";

type Statuses = ReturnType<typeof useConnectors>;
type Host = "github" | "gitlab";

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

const PLACEHOLDER: Record<Host, string> = { github: "owner/repo", gitlab: "group/project" };

/** What is being typed: a new item (index null) or an edit of an existing one. */
interface Draft {
  name: ConnectorName;
  index: number | null;
  path: string;
  ref: string;
}

/** The Jira account being connected: where, as whom, and the project issues go to. */
interface AccountDraft {
  site: string;
  project: string;
  email: string;
  token: string;
}

const TOKENS_URL = "https://id.atlassian.com/manage-profile/security/api-tokens";
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

function State({ name, statuses }: { name: ConnectorName; statuses: Statuses }) {
  const status = statuses.data?.find((c) => c.name === name);
  const state = status?.state ?? (statuses.error ? "failing" : null);
  return (
    <span className="conn-state" data-state={state ?? "checking"} title={status?.detail ?? (statuses.error ? describeError(statuses.error) : undefined)}>
      {state ? STATE[state] : "Checking…"}
    </span>
  );
}

/** One row per connected repository, project and Jira; admins add, edit and remove them inline.
 * Each change is saved at once (PUT /settings/connectors) and the states are checked again.
 * The Jira account approved tasks are pushed to is a row of its own, with its own site and
 * project: connecting it leaves the Jira project the agent reads as it is. */
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
  const [account, setAccount] = useState<AccountDraft | null>(null);
  const [adding, setAdding] = useState(false);
  const [menu, setMenu] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [problem, setProblem] = useState("");
  const addRef = useRef<HTMLDivElement>(null);
  const rowMenu = useRef<HTMLDivElement>(null);
  useDismiss(addRef, adding, useCallback(() => setAdding(false), []));
  useDismiss(rowMenu, menu !== null, useCallback(() => setMenu(null), []));

  const save = async (next: ConnectorsUpdate) => {
    setBusy(true);
    setProblem("");
    try {
      onSaved(await updateConnectors(next));
      setDraft(null);
      statuses.reload();
    } catch (err) {
      setProblem(`Not saved. ${describeError(err)}`);
    } finally {
      setBusy(false);
    }
  };

  const startAccount = () => {
    setMenu(null);
    setDraft(null);
    setProblem("");
    setAccount({
      site: settings.jira.account_site ?? "",
      project: settings.jira.account_project ?? "",
      email: settings.jira.account_email ?? "",
      token: "",
    });
  };

  const connectAccount = async (e: FormEvent) => {
    e.preventDefault();
    if (!account) return;
    setBusy(true);
    setProblem("");
    try {
      onSaved(
        await connectJiraAccount({
          site: account.site.trim(),
          project: account.project.trim().toUpperCase(),
          email: account.email.trim(),
          api_token: account.token.trim(),
        }),
      );
      setAccount(null);
      statuses.reload();
    } catch (err) {
      setProblem(`Not connected. ${describeError(err)}`);
    } finally {
      setBusy(false);
    }
  };

  const disconnectAccount = async () => {
    setMenu(null);
    setBusy(true);
    setProblem("");
    try {
      onSaved(await disconnectJiraAccount());
      statuses.reload();
    } catch (err) {
      setProblem(`Not disconnected. ${describeError(err)}`);
    } finally {
      setBusy(false);
    }
  };

  const remove = (name: ConnectorName, index: number) => {
    setMenu(null);
    const next = current(settings);
    if (name === "jira") next.jira = { site: null, project: null };
    else next[name] = next[name].filter((_, i) => i !== index);
    void save(next);
  };

  const submit = (e: FormEvent) => {
    e.preventDefault();
    if (!draft) return;
    const next = current(settings);
    const path = draft.path.trim();
    const ref = draft.ref.trim() || null;
    if (draft.name === "jira") {
      next.jira = { site: ref, project: path.toUpperCase() || null };
    } else if (draft.index === null) {
      next[draft.name] = [...next[draft.name], { path, ref }];
    } else {
      next[draft.name] = next[draft.name].map((r, i) => (i === draft.index ? { path, ref } : r));
    }
    void save(next);
  };

  const start = (name: ConnectorName, index: number | null = null) => {
    setAdding(false);
    setMenu(null);
    setAccount(null);
    setProblem("");
    if (name === "jira") {
      setDraft({ name, index: 0, path: settings.jira.project ?? "", ref: settings.jira.site ?? "" });
      return;
    }
    const repo = index === null ? null : (name === "github" ? settings.github.repos : settings.gitlab.projects)[index];
    setDraft({ name, index, path: repo?.path ?? "", ref: repo?.ref ?? "" });
  };

  const rows: { name: ConnectorName; index: number; label: string; detail: string | null }[] = [
    ...settings.github.repos.map((r, index) => ({ name: "github" as const, index, label: r.path, detail: r.ref ?? null })),
    ...settings.gitlab.projects.map((r, index) => ({ name: "gitlab" as const, index, label: r.path, detail: r.ref ?? null })),
    ...(settings.jira.project
      ? [
          { name: "jira" as const, index: 0, label: settings.jira.project, detail: settings.jira.site ?? null },
        ]
      : []),
  ];
  const full: Record<ConnectorName, boolean> = {
    github: settings.github.repos.length >= MAX_CODE_REPOS,
    gitlab: settings.gitlab.projects.length >= MAX_CODE_REPOS,
    jira: Boolean(settings.jira.project),
  };

  const form = (d: Draft) => (
    <form className="conn-row conn-edit" onSubmit={submit}>
      <Icon name={d.name} className={`conn-icon is-${d.name}`} />
      <input
        className="field mono"
        aria-label={d.name === "jira" ? "Jira project key" : d.name === "github" ? "GitHub repository" : "GitLab project"}
        value={d.path}
        onChange={(e) => setDraft({ ...d, path: e.target.value })}
        placeholder={d.name === "jira" ? "KEY" : PLACEHOLDER[d.name]}
        style={d.name === "jira" ? { textTransform: "uppercase" } : undefined}
        autoFocus
        required
      />
      <input
        className="field mono conn-ref"
        aria-label={d.name === "jira" ? "Jira site" : "Branch or tag"}
        value={d.ref}
        onChange={(e) => setDraft({ ...d, ref: e.target.value })}
        placeholder={d.name === "jira" ? "team.atlassian.net" : "branch"}
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

  return (
    <div className="conn">
      <ul className="conn-list">
        {rows.map((row) => {
          const key = `${row.name}:${row.index}`;
          if (draft && draft.name === row.name && draft.index === row.index) return <li key={key}>{form(draft)}</li>;
          return (
            <li key={key} className="conn-row">
              <Icon name={row.name} className={`conn-icon is-${row.name}`} />
              <span className="conn-name">
                {row.label}
                {row.detail && <span className="conn-ref-text"> · {row.detail}</span>}
              </span>
              <State name={row.name} statuses={statuses} />
              {canEdit && (
                <div className="conn-more" ref={menu === key ? rowMenu : undefined}>
                  <button
                    type="button"
                    className="icon-btn sm"
                    aria-label={`More for ${row.label}`}
                    aria-haspopup="menu"
                    aria-expanded={menu === key}
                    onClick={() => setMenu(menu === key ? null : key)}
                    disabled={busy}
                  >
                    <Icon name="ellipsis" />
                  </button>
                  {menu === key && (
                    <div className="menu conn-menu" role="menu" aria-label={row.label}>
                      <button type="button" className="menu-item" role="menuitem" onClick={() => start(row.name, row.index)}>
                        <Icon name="pencil" />
                        Edit
                      </button>

                      <button type="button" className="menu-item danger" role="menuitem" onClick={() => remove(row.name, row.index)}>
                        <Icon name="trash-2" />
                        Remove
                      </button>
                    </div>
                  )}
                </div>
              )}
            </li>
          );
        })}
        {draft && draft.index === null && <li>{form(draft)}</li>}
        {draft?.name === "jira" && !settings.jira.project && <li>{form(draft)}</li>}
        {settings.jira.connected && !account && (
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
            {canEdit && (
              <div className="conn-more" ref={menu === ACCOUNT ? rowMenu : undefined}>
                <button
                  type="button"
                  className="icon-btn sm"
                  aria-label="More for the Jira account"
                  aria-haspopup="menu"
                  aria-expanded={menu === ACCOUNT}
                  onClick={() => setMenu(menu === ACCOUNT ? null : ACCOUNT)}
                  disabled={busy}
                >
                  <Icon name="ellipsis" />
                </button>
                {menu === ACCOUNT && (
                  <div className="menu conn-menu" role="menu" aria-label="Jira account">
                    <button type="button" className="menu-item" role="menuitem" onClick={startAccount}>
                      <Icon name="pencil" />
                      Change
                    </button>
                    <button type="button" className="menu-item danger" role="menuitem" onClick={() => void disconnectAccount()}>
                      <Icon name="trash-2" />
                      Disconnect
                    </button>
                  </div>
                )}
              </div>
            )}
          </li>
        )}
        {account && (
          <li>
            <form className="conn-account" onSubmit={(e) => void connectAccount(e)} aria-label="Connect a Jira account">
              <h3>Jira account for pushing tasks</h3>
              <label className="label">
                Jira site
                <input
                  className="field mono"
                  value={account.site}
                  onChange={(e) => setAccount({ ...account, site: e.target.value })}
                  placeholder="your-team.atlassian.net"
                  autoFocus
                  required
                />
              </label>
              <label className="label">
                Project key
                <input
                  className="field mono"
                  value={account.project}
                  onChange={(e) => setAccount({ ...account, project: e.target.value })}
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
                  value={account.email}
                  onChange={(e) => setAccount({ ...account, email: e.target.value })}
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
                  value={account.token}
                  onChange={(e) => setAccount({ ...account, token: e.target.value })}
                  autoComplete="new-password"
                  required
                />
              </label>
              <p className="note">
                Approved tasks are created in this project as this account; what the assistant reads from Jira is not
                changed. Create a token at{" "}
                <a className="link" href={TOKENS_URL} target="_blank" rel="noopener noreferrer">
                  id.atlassian.com
                </a>{" "}
                (Security, API tokens). It is checked with Jira, stored encrypted and never shown again.
              </p>
              <div className="conn-account-actions">
                <button
                  type="submit"
                  className="btn btn-primary btn-sm"
                  disabled={busy || !account.site.trim() || !account.project.trim() || !account.email.trim() || !account.token.trim()}
                >
                  {busy ? <Icon name="loader-circle" className="spin" /> : <Icon name="link" />}
                  Connect
                </button>
                <button type="button" className="btn btn-quiet btn-sm" onClick={() => setAccount(null)} disabled={busy}>
                  Cancel
                </button>
              </div>
            </form>
          </li>
        )}
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
              <button
                type="button"
                className="menu-item"
                role="menuitem"
                disabled={settings.jira.connected}
                onClick={() => {
                  setAdding(false);
                  startAccount();
                }}
              >
                <Icon name="jira" className="conn-icon is-jira" />
                Jira account, to push tasks
              </button>
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
