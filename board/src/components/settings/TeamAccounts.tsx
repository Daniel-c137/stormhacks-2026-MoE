"use client";

import type { CreateAccountResponse } from "@moe/contracts";
import { type FormEvent, useState } from "react";
import { Avatar } from "@/components/ui/Avatar";
import { Icon, Spinner } from "@/components/ui/Icon";
import { Notice } from "@/components/ui/Notice";
import { useMembers } from "@/hooks/useApi";
import { ApiError, createAccount, describeError } from "@/lib/api";

type Role = "member" | "admin";

/** Admins only: the team's members and adding a person (POST /team/accounts). The server makes
 * the password; it shows here once and is never sent anywhere else. */
export function TeamAccounts() {
  const members = useMembers();
  const [adding, setAdding] = useState(false);
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [role, setRole] = useState<Role>("member");
  const [busy, setBusy] = useState(false);
  const [problem, setProblem] = useState("");
  const [created, setCreated] = useState<CreateAccountResponse | null>(null);
  const [copied, setCopied] = useState(false);

  const close = () => {
    setAdding(false);
    setName("");
    setEmail("");
    setRole("member");
    setProblem("");
  };

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setProblem("");
    try {
      setCreated(await createAccount({ name, email, is_admin: role === "admin" }));
      setCopied(false);
      close();
      members.reload();
    } catch (err) {
      setProblem(err instanceof ApiError && err.status === 409 ? "Someone already signs in with this email." : describeError(err));
    } finally {
      setBusy(false);
    }
  };

  const copy = async () => {
    if (!created) return;
    setProblem("");
    try {
      await navigator.clipboard.writeText(created.password);
      setCopied(true);
    } catch {
      // The Clipboard API can be refused (an embedded or unfocused page); copy the selected text.
      const code = document.getElementById("new-password");
      const selection = window.getSelection();
      if (code && selection) {
        selection.selectAllChildren(code);
        if (document.execCommand("copy")) return setCopied(true);
      }
      setProblem("Couldn't copy. Select the password and copy it.");
    }
  };

  return (
    <div className="sgroup" role="group" aria-labelledby="g-members">
      <div className="members-head">
        <h2 id="g-members">Members</h2>
        {!adding && (
          <button
            type="button"
            className="btn btn-outline btn-sm"
            onClick={() => {
              setAdding(true);
              setCreated(null);
            }}
          >
            <Icon name="plus" />
            Add person
          </button>
        )}
      </div>
      {created && (
        <div className="once" role="status">
          <span>
            One-time password for <b>{created.person.name}</b>
          </span>
          <code id="new-password">{created.password}</code>
          <button type="button" className="btn btn-outline btn-sm" onClick={() => void copy()}>
            <Icon name={copied ? "check" : "copy"} />
            {copied ? "Copied" : "Copy"}
          </button>
          <button type="button" className="icon-btn sm" onClick={() => setCreated(null)} aria-label="Dismiss password">
            <Icon name="x" />
          </button>
        </div>
      )}
      {adding && (
        <form className="member-add" onSubmit={submit} aria-label="Add person">
          <input
            className="field"
            placeholder="Name"
            aria-label="Name"
            value={name}
            onChange={(e) => setName(e.target.value)}
            autoComplete="off"
            autoFocus
            required
          />
          <input
            className="field"
            type="email"
            placeholder="Email"
            aria-label="Email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            autoComplete="off"
            required
          />
          <select className="field" aria-label="Role" value={role} onChange={(e) => setRole(e.target.value as Role)}>
            <option value="member">Member</option>
            <option value="admin">Admin</option>
          </select>
          <div className="member-add-acts">
            <button type="button" className="btn btn-quiet btn-sm" onClick={close}>
              Cancel
            </button>
            <button type="submit" className="btn btn-primary btn-sm" disabled={busy || !name.trim() || !email.trim()}>
              {busy && <Spinner />}
              Add
            </button>
          </div>
        </form>
      )}
      {problem && (
        <p className="err member-msg" role="alert">
          {problem}
        </p>
      )}
      {members.data ? (
        <ul className="members">
          {members.data.map((p) => (
            <li key={p.id} className="member">
              <Avatar person={p} />
              <span className="member-text">
                <span className="member-name">{p.name}</span>
                {p.email && <span className="member-email">{p.email}</span>}
              </span>
              {p.is_admin ? <span className="badge">Admin</span> : <span className="member-role">Member</span>}
            </li>
          ))}
        </ul>
      ) : members.error ? (
        <Notice error={members.error} onRetry={members.reload}>
          Members can&apos;t be loaded.
        </Notice>
      ) : (
        <p className="note member-msg">Loading…</p>
      )}
    </div>
  );
}
