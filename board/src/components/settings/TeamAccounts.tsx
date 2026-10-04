"use client";

import type { CreateAccountResponse } from "@moe/contracts";
import { type FormEvent, useState } from "react";
import { Avatar } from "@/components/ui/Avatar";
import { Icon, Spinner } from "@/components/ui/Icon";
import { Notice } from "@/components/ui/Notice";
import { useMembers } from "@/hooks/useApi";
import { createAccount, describeError } from "@/lib/api";

type Role = "member" | "admin";
/** Invite: no login; the person creates their account on the sign-in page. Password: the server
 * makes one, shown here once. */
type How = "invite" | "password";

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

/** Admins only: the team's members and adding a person (POST /team/accounts), invited or with a
 * generated password. The password shows here once and is never sent anywhere else. */
export function TeamAccounts() {
  const members = useMembers();
  const [adding, setAdding] = useState(false);
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [role, setRole] = useState<Role>("member");
  const [how, setHow] = useState<How>("invite");
  const [busy, setBusy] = useState(false);
  const [problem, setProblem] = useState("");
  const [created, setCreated] = useState<CreateAccountResponse | null>(null);
  const [copied, setCopied] = useState(false);

  const close = () => {
    setAdding(false);
    setName("");
    setEmail("");
    setRole("member");
    setHow("invite");
    setProblem("");
  };

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setProblem("");
    try {
      setCreated(await createAccount({ name, email, is_admin: role === "admin", invite: how === "invite" }));
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

  /** The password, or the sign-up link for an invite. */
  const copy = async () => {
    if (!created) return;
    setProblem("");
    try {
      await navigator.clipboard.writeText(created.password ?? signUpLink());
      setCopied(true);
    } catch {
      // The Clipboard API can be refused (an embedded or unfocused page); copy the selected text.
      if (!created.password) {
        if (copySelected(signUpLink())) return setCopied(true);
        return setProblem(`Couldn't copy. The link is ${signUpLink()}`);
      }
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
          {created.password ? (
            <>
              <span>
                One-time password for <b>{created.person.name}</b>
              </span>
              <code id="new-password">{created.password}</code>
            </>
          ) : (
            <span>
              Invited <b>{created.person.name}</b>.
            </span>
          )}
          <button type="button" className="btn btn-outline btn-sm" onClick={() => void copy()}>
            <Icon name={copied ? "check" : created.password ? "copy" : "link"} />
            {copied ? "Copied" : created.password ? "Copy" : "Copy link"}
          </button>
          <button type="button" className="icon-btn sm" onClick={() => setCreated(null)} aria-label={created.password ? "Dismiss password" : "Dismiss"}>
            <Icon name="x" />
          </button>
        </div>
      )}
      {adding && (
        <form className="member-add" onSubmit={submit} aria-label="Add person">
          <div className="seg" role="radiogroup" aria-label="How they sign in">
            <button type="button" role="radio" aria-checked={how === "invite"} onClick={() => setHow("invite")}>
              <Icon name="link" />
              Invite
            </button>
            <button type="button" role="radio" aria-checked={how === "password"} onClick={() => setHow("password")}>
              <Icon name="lock" />
              Generate password
            </button>
          </div>
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
              {how === "invite" ? "Invite" : "Add"}
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
