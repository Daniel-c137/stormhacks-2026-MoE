"use client";

import type { CreateAccountResponse } from "@moe/contracts";
import { type FormEvent, useState } from "react";
import { Avatar } from "@/components/ui/Avatar";
import { Icon, Spinner } from "@/components/ui/Icon";
import { Notice } from "@/components/ui/Notice";
import { useMembers } from "@/hooks/useApi";
import { createAccount, describeError } from "@/lib/api";

type Role = "member" | "admin";

/** The sign-in page opened on Create account, to send to someone invited. */
const signUpLink = () => `${window.location.origin}/login?mode=signup`;

/** Copies `text` through a selection, for when the Clipboard API is refused. */
function copySelected(text: string): boolean {
  const box = document.createElement("textarea");
  box.value = text;
  box.setAttribute("readonly", "");
  box.style.position = "fixed";
  box.style.opacity = "0";
  document.body.appendChild(box);
  box.select();
  try {
    return document.execCommand("copy");
  } finally {
    box.remove();
  }
}

/** Admins only: the team's members, and inviting someone by email (POST /team/accounts with
 * invite). They create their own account on the sign-in page with that email, choosing their name
 * and password, or with Google. */
export function TeamAccounts() {
  const members = useMembers();
  const [adding, setAdding] = useState(false);
  const [email, setEmail] = useState("");
  const [role, setRole] = useState<Role>("member");
  const [busy, setBusy] = useState(false);
  const [problem, setProblem] = useState("");
  const [invited, setInvited] = useState<CreateAccountResponse | null>(null);
  const [copied, setCopied] = useState(false);

  const close = () => {
    setAdding(false);
    setEmail("");
    setRole("member");
    setProblem("");
  };

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setProblem("");
    try {
      setInvited(await createAccount({ email, is_admin: role === "admin", invite: true }));
      setCopied(false);
      close();
      members.reload();
    } catch (err) {
      // 409: on this team already, has an account, or another team invited it; the brain says which.
      setProblem(describeError(err));
    } finally {
      setBusy(false);
    }
  };

  const copy = async () => {
    setProblem("");
    try {
      await navigator.clipboard.writeText(signUpLink());
      setCopied(true);
    } catch {
      // The Clipboard API can be refused (an embedded or unfocused page); copy the selected text.
      if (copySelected(signUpLink())) return setCopied(true);
      setProblem(`Couldn't copy. The link is ${signUpLink()}`);
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
              setInvited(null);
            }}
          >
            <Icon name="plus" />
            Invite
          </button>
        )}
      </div>
      {invited && (
        <div className="once" role="status">
          <span>
            Invited <b>{invited.person.email}</b>
          </span>
          <button type="button" className="btn btn-outline btn-sm" onClick={() => void copy()}>
            <Icon name={copied ? "check" : "link"} />
            {copied ? "Copied" : "Copy link"}
          </button>
          <button type="button" className="icon-btn sm" onClick={() => setInvited(null)} aria-label="Dismiss">
            <Icon name="x" />
          </button>
        </div>
      )}
      {adding && (
        <form className="member-add" onSubmit={submit} aria-label="Invite">
          <input
            className="field"
            type="email"
            placeholder="Email"
            aria-label="Email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            autoComplete="off"
            autoFocus
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
            <button type="submit" className="btn btn-primary btn-sm" disabled={busy || !email.trim()}>
              {busy && <Spinner />}
              Invite
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
              {p.invited ? (
                <span className="badge badge-invited">Invited</span>
              ) : p.is_admin ? (
                <span className="badge">Admin</span>
              ) : (
                <span className="member-role">Member</span>
              )}
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
