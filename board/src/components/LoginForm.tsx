"use client";

import { type AuthOptions, identity } from "@moe/contracts";
import { useRouter } from "next/navigation";
import { type FormEvent, useEffect, useState } from "react";
import { useAuth } from "@/components/AuthProvider";
import { waitText } from "@/lib/api";
import { apiBase, apiConfigured } from "@/lib/apiUrl";
import { LoginError, authOptions, finishGoogleSignIn, googleSignInUrl, login, signUp } from "@/lib/auth";
import { safeNextPath } from "@/lib/nextPath";
import styles from "./LoginForm.module.css";

/** Sign in, create an account (invited emails only) and Google, all checked by the brain
 * (lib/auth.ts). The design is Reyhaneh's (#117), without Supabase (#128). */

type Mode = "signin" | "signup";
type Notice = { kind: "error" | "info"; text: string } | null;

/** The brain's own rule (brain/auth.py MIN_PASSWORD). */
const MIN_PASSWORD = 10;

/** Why the brain's Google callback sent the browser back without a session. */
const GOOGLE_ERRORS: Record<string, string> = {
  cancelled: "Google sign-in was cancelled.",
  state: "That sign-in didn't start in this browser. Try again.",
  not_invited: "That Google account hasn't been invited. Ask your team's admin to add you.",
  unverified: "Google hasn't verified that email address.",
  failed: "Google sign-in didn't work. Try again.",
};

/** Where to go after auth, also after Google: the ?next= path when it stays on this site, else
 * home. */
function nextPath(): string {
  if (typeof window === "undefined") return "/";
  return safeNextPath(new URLSearchParams(window.location.search).get("next"), window.location.origin);
}

function authProblem(err: unknown, mode: Mode | "google"): string {
  if (err instanceof LoginError) {
    if (err.status === 401) return mode === "google" ? GOOGLE_ERRORS.failed : "That email and password don't match an account.";
    if (err.status === 403) return "That email hasn't been invited. Ask your team's admin to add you.";
    if (err.status === 409) return "An account already uses that email. Sign in instead.";
    if (err.status === 429)
      return err.retryAfter ? `Too many failed attempts. Try again in ${waitText(err.retryAfter)}.` : "Too many failed attempts. Try again later.";
    if (err.status === 503) return `This isn't available right now: ${err.message}`;
    return err.message;
  }
  if (err instanceof TypeError) return `Can't reach the server${apiBase() ? ` at ${apiBase()}` : ""}.`;
  return err instanceof Error ? err.message : "Something went wrong. Try again.";
}

export function LoginForm() {
  const status = useAuth();
  const router = useRouter();
  const [mode, setMode] = useState<Mode>("signin");
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState<Notice>(null);
  const [offers, setOffers] = useState<AuthOptions>({ signup: false, google: false });

  const [fullName, setFullName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");

  useEffect(() => {
    if (status === "signed_in") router.replace(nextPath());
  }, [status, router]);

  // What the brain offers; /login?mode=signup opens Create account; the Google callback lands
  // here with ?google=<one-time code> or ?google_error=<why>.
  useEffect(() => {
    if (!apiConfigured()) return;
    void authOptions().then(setOffers);
    const query = new URLSearchParams(window.location.search);
    if (query.get("mode") === "signup") setMode("signup");
    const failed = query.get("google_error");
    if (failed) setNotice({ kind: "error", text: GOOGLE_ERRORS[failed] ?? GOOGLE_ERRORS.failed });
    const code = query.get("google");
    if (code) {
      setBusy(true);
      finishGoogleSignIn(code).catch((err: unknown) => {
        setNotice({ kind: "error", text: authProblem(err, "google") });
        setBusy(false);
      });
      // The session change moves us on (above).
    }
  }, []);

  const switchMode = (next: Mode) => {
    setMode(next);
    setNotice(null);
    setPassword("");
    setConfirm("");
  };

  const isSignUp = mode === "signup" && offers.signup;
  const passwordTooShort = isSignUp && password.length > 0 && password.length < MIN_PASSWORD;
  const confirmState = confirm.length === 0 ? null : confirm === password ? "match" : "mismatch";

  async function run(action: () => Promise<unknown>) {
    setBusy(true);
    setNotice(null);
    try {
      await action();
      // The session change moves us on (above).
    } catch (err) {
      setNotice({ kind: "error", text: authProblem(err, mode) });
      setBusy(false);
    }
  }

  const onGoogle = () => {
    setBusy(true);
    window.location.assign(googleSignInUrl(nextPath()));
  };

  const onSignIn = (e: FormEvent) => {
    e.preventDefault();
    if (!email.trim() || !password) {
      setNotice({ kind: "error", text: "Enter your email and password." });
      return;
    }
    void run(() => login(email.trim(), password));
  };

  const onSignUp = (e: FormEvent) => {
    e.preventDefault();
    if (!fullName.trim() || !email.trim() || !password || !confirm) {
      setNotice({ kind: "error", text: "Please fill in every field." });
      return;
    }
    if (password.length < MIN_PASSWORD) {
      setNotice({ kind: "error", text: `Password must be at least ${MIN_PASSWORD} characters.` });
      return;
    }
    if (password !== confirm) {
      setNotice({ kind: "error", text: "Passwords don't match." });
      return;
    }
    void run(() => signUp(fullName.trim(), email.trim(), password));
  };

  const onForgot = () =>
    setNotice({ kind: "info", text: `Your team's admin can reset your ${identity.product_name} password.` });

  if (!apiConfigured()) {
    return (
      <div className={styles.page}>
        <main className={styles.card}>
          <Brand />
          <p className={styles.noticeError} role="status">
            Sign-in isn&apos;t configured. Set NEXT_PUBLIC_API_URL in the repo&apos;s .env and restart.
          </p>
        </main>
      </div>
    );
  }

  return (
    <div className={styles.page}>
      <main className={styles.card}>
        <Brand />

        {offers.signup && (
          <div className={styles.tabs} role="tablist" aria-label="Sign in or create an account">
            <button type="button" role="tab" className={styles.tab} aria-selected={!isSignUp} onClick={() => switchMode("signin")}>
              Sign in
            </button>
            <button type="button" role="tab" className={styles.tab} aria-selected={isSignUp} onClick={() => switchMode("signup")}>
              Create account
            </button>
          </div>
        )}

        <h1 className={styles.title}>{isSignUp ? "Create account" : "Sign in"}</h1>

        {offers.google && (
          <>
            <button type="button" className={styles.google} onClick={onGoogle} disabled={busy}>
              <GoogleIcon />
              <span>{isSignUp ? "Sign up with Google" : "Sign in with Google"}</span>
            </button>
            <div className={styles.divider}>or</div>
          </>
        )}

        {isSignUp ? (
          <form className={styles.form} onSubmit={onSignUp} noValidate>
            <div>
              <label className={styles.label} htmlFor="su-name">
                Full name
              </label>
              <input id="su-name" className={styles.input} type="text" autoComplete="name" value={fullName} onChange={(e) => setFullName(e.target.value)} />
            </div>
            <div>
              <label className={styles.label} htmlFor="su-email">
                Email
              </label>
              <input id="su-email" className={styles.input} type="email" autoComplete="email" value={email} onChange={(e) => setEmail(e.target.value)} />
              <p className={styles.hint}>The email your team&apos;s admin invited.</p>
            </div>
            <div>
              <label className={styles.label} htmlFor="su-password">
                Password
              </label>
              <input
                id="su-password"
                className={`${styles.input} ${passwordTooShort ? styles.invalid : ""}`}
                type="password"
                autoComplete="new-password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
              />
              <p className={`${styles.hint} ${passwordTooShort ? styles.hintError : ""}`}>At least {MIN_PASSWORD} characters.</p>
            </div>
            <div>
              <label className={styles.label} htmlFor="su-confirm">
                Confirm password
              </label>
              <input
                id="su-confirm"
                className={`${styles.input} ${confirmState === "mismatch" ? styles.invalid : ""}`}
                type="password"
                autoComplete="new-password"
                value={confirm}
                onChange={(e) => setConfirm(e.target.value)}
              />
              {confirmState && (
                <p className={`${styles.hint} ${confirmState === "match" ? styles.hintOk : styles.hintError}`}>
                  {confirmState === "match" ? "Passwords match." : "Passwords don't match."}
                </p>
              )}
            </div>
            <button type="submit" className={styles.primary} disabled={busy}>
              <UserPlusIcon />
              {busy ? "Creating account…" : "Create account"}
            </button>
          </form>
        ) : (
          <form className={styles.form} onSubmit={onSignIn} noValidate>
            <div>
              <label className={styles.label} htmlFor="si-email">
                Email
              </label>
              <input id="si-email" className={styles.input} type="email" autoComplete="email" value={email} onChange={(e) => setEmail(e.target.value)} />
            </div>
            <div>
              <div className={styles.rowBetween}>
                <label className={styles.label} htmlFor="si-password">
                  Password
                </label>
                <button type="button" className={styles.link} onClick={onForgot} disabled={busy}>
                  Forgot password?
                </button>
              </div>
              <input
                id="si-password"
                className={styles.input}
                type="password"
                autoComplete="current-password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
              />
            </div>
            <button type="submit" className={styles.primary} disabled={busy}>
              <SignInIcon />
              {busy ? "Signing in…" : "Sign in"}
            </button>
          </form>
        )}

        {notice && (
          <p className={notice.kind === "error" ? styles.noticeError : styles.noticeInfo} role="status">
            {notice.text}
          </p>
        )}

        {offers.signup ? (
          <p className={styles.switch}>
            {isSignUp ? "Already have an account? " : "Been invited? "}
            <button type="button" onClick={() => switchMode(isSignUp ? "signin" : "signup")}>
              {isSignUp ? "Sign in" : "Create your account"}
            </button>
          </p>
        ) : (
          <p className={styles.switch}>No account? Your team&apos;s admin creates accounts for {identity.product_name}.</p>
        )}
      </main>
    </div>
  );
}

function Brand() {
  return (
    <div className={styles.brand}>
      <svg viewBox="-50 -50 100 100" aria-hidden="true">
        <path d="M0 -46 Q5 -5 40 0 Q5 5 0 46 Q-5 5 -40 0 Q-5 -5 0 -46Z" fill="none" stroke="currentColor" strokeWidth="3" strokeLinejoin="round" />
        <circle r="3.5" fill="currentColor" />
      </svg>
      <span>{identity.product_name}</span>
    </div>
  );
}

function GoogleIcon() {
  return (
    <svg viewBox="0 0 48 48" aria-hidden="true">
      <path
        fill="#FFC107"
        d="M43.6 20.5H42V20H24v8h11.3C33.7 32.7 29.2 36 24 36c-6.6 0-12-5.4-12-12s5.4-12 12-12c3.1 0 5.8 1.2 7.9 3.1l5.7-5.7C34 6.1 29.3 4 24 4 12.9 4 4 12.9 4 24s8.9 20 20 20 20-8.9 20-20c0-1.3-.1-2.4-.4-3.5z"
      />
      <path fill="#FF3D00" d="M6.3 14.7l6.6 4.8C14.7 15.1 19 12 24 12c3.1 0 5.8 1.2 7.9 3.1l5.7-5.7C34 6.1 29.3 4 24 4 16.3 4 9.7 8.3 6.3 14.7z" />
      <path fill="#4CAF50" d="M24 44c5.2 0 9.9-2 13.4-5.2l-6.2-5.2C29.2 35.1 26.7 36 24 36c-5.2 0-9.6-3.3-11.3-8l-6.5 5C9.5 39.6 16.2 44 24 44z" />
      <path fill="#1976D2" d="M43.6 20.5H42V20H24v8h11.3c-.8 2.2-2.2 4.2-4.1 5.6l6.2 5.2C37 39.2 44 34 44 24c0-1.3-.1-2.4-.4-3.5z" />
    </svg>
  );
}

function SignInIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M15 3h4a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2h-4" />
      <path d="M10 17l5-5-5-5" />
      <path d="M15 12H3" />
    </svg>
  );
}

function UserPlusIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2" />
      <circle cx="9" cy="7" r="4" />
      <path d="M19 8v6" />
      <path d="M22 11h-6" />
    </svg>
  );
}
