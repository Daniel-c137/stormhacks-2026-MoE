"use client";

import { identity } from "@moe/contracts";
import { useRouter } from "next/navigation";
import { type FormEvent, useEffect, useState } from "react";
import { useAuth } from "@/components/AuthProvider";
import { Icon, Spinner } from "@/components/ui/Icon";
import { Mark } from "@/components/ui/Mark";
import { supabaseBrowser } from "@/lib/supabase";

/** Where to land after sign-in: the page that sent us here, if it is one of ours. */
function nextPath(): string {
  const next = new URLSearchParams(window.location.search).get("next");
  return next && next.startsWith("/") && !next.startsWith("//") ? next : "/";
}

/** Supabase email/password sign-in and sign-up. */
export function LoginForm() {
  const auth = useAuth();
  const router = useRouter();
  const [mode, setMode] = useState<"in" | "up">("in");
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [note, setNote] = useState("");

  useEffect(() => {
    if (auth.status === "signed_in") router.replace(nextPath());
  }, [auth.status, router]);

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setError("");
    setNote("");
    try {
      const sb = supabaseBrowser().auth;
      if (mode === "in") {
        const { error: err } = await sb.signInWithPassword({ email, password });
        if (err) throw err;
      } else {
        const { data, error: err } = await sb.signUp({ email, password, options: { data: { name: name.trim() } } });
        if (err) throw err;
        if (!data.session) setNote("Check your email to confirm your account, then sign in.");
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Sign-in failed.");
    } finally {
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
        <h1 className="auth-title">{mode === "in" ? "Sign in" : "Create your account"}</h1>
        {auth.status === "unconfigured" ? (
          <p className="notice">
            <Icon name="info" />
            <span>
              Sign-in isn&apos;t configured. Set NEXT_PUBLIC_SUPABASE_URL and NEXT_PUBLIC_SUPABASE_ANON_KEY in the
              repo&apos;s .env and restart.
            </span>
          </p>
        ) : (
          <>
            <form onSubmit={submit}>
              {mode === "up" && (
                <label className="label">
                  Name
                  <input className="field" value={name} onChange={(e) => setName(e.target.value)} autoComplete="name" required />
                </label>
              )}
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
                  autoComplete={mode === "in" ? "current-password" : "new-password"}
                  minLength={6}
                  required
                />
              </label>
              <span role="alert" className="err">
                {error}
              </span>
              {note && <p className="note">{note}</p>}
              <button type="submit" className="btn btn-primary" disabled={busy}>
                {busy ? <Spinner /> : <Icon name="log-in" />}
                {mode === "in" ? "Sign in" : "Sign up"}
              </button>
            </form>
            <p className="auth-switch">
              {mode === "in" ? "New here? " : "Already have an account? "}
              <button type="button" className="link" onClick={() => setMode(mode === "in" ? "up" : "in")}>
                {mode === "in" ? "Create an account" : "Sign in"}
              </button>
            </p>
          </>
        )}
      </div>
    </main>
  );
}
