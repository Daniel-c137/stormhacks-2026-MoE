"use client";

import { identity } from "@moe/contracts";
import { useRouter } from "next/navigation";
import { type FormEvent, useEffect, useState } from "react";
import { useAuth } from "@/components/AuthProvider";
import { Icon, Spinner } from "@/components/ui/Icon";
import { Mark } from "@/components/ui/Mark";
import { waitText } from "@/lib/api";
import { apiBase, apiConfigured } from "@/lib/apiUrl";
import { LoginError, login } from "@/lib/auth";

/** Where to land after sign-in: the page that sent us here, if it is one of ours. */
function nextPath(): string {
  const next = new URLSearchParams(window.location.search).get("next");
  return next && next.startsWith("/") && !next.startsWith("//") ? next : "/";
}

function loginProblem(err: unknown): string {
  if (err instanceof LoginError) {
    if (err.status === 401) return "That email and password don't match an account.";
    if (err.status === 429)
      return err.retryAfter
        ? `Too many failed attempts. Try again in ${waitText(err.retryAfter)}.`
        : "Too many failed attempts. Try again later.";
    if (err.status === 503) return `Sign-in isn't available right now: ${err.message}`;
    return err.message;
  }
  if (err instanceof TypeError) return `Can't reach the server${apiBase() ? ` at ${apiBase()}` : ""}.`;
  return err instanceof Error ? err.message : "Sign-in failed.";
}

/** Email/password sign-in, checked by the brain (lib/auth.ts, POST /auth/login). There is no
 * sign-up: accounts come from the team's admin. */
export function LoginForm() {
  const status = useAuth();
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    if (status === "signed_in") router.replace(nextPath());
  }, [status, router]);

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setError("");
    try {
      await login(email.trim(), password);
      // The session change moves us on (above).
    } catch (err) {
      setError(loginProblem(err));
      setBusy(false);
    }
  };

  return (
    <main className="auth">
      <div className="auth-card">
        <span className="brand">
          <span className="mark-slot">
            <Mark size={30} />
          </span>
          <span className="wordmark">{identity.product_name}</span>
        </span>
        <h1 className="auth-title">Sign in</h1>
        {!apiConfigured() ? (
          <p className="notice">
            <Icon name="info" />
            <span>Sign-in isn&apos;t configured. Set NEXT_PUBLIC_API_URL in the repo&apos;s .env and restart.</span>
          </p>
        ) : (
          <>
            <form onSubmit={submit}>
              <label className="label">
                Email
                <input
                  className="field"
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  autoComplete="email"
                  required
                />
              </label>
              <label className="label">
                Password
                <input
                  className="field"
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  autoComplete="current-password"
                  required
                />
              </label>
              <span role="alert" className="err">
                {error}
              </span>
              <button type="submit" className="btn btn-primary" disabled={busy}>
                {busy ? <Spinner /> : <Icon name="log-in" />}
                Sign in
              </button>
            </form>
            <p className="auth-switch">No account? Your team&apos;s admin creates accounts for {identity.product_name}.</p>
          </>
        )}
      </div>
    </main>
  );
}
