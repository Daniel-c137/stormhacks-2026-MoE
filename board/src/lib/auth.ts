// Sign-in against the brain: POST /auth/login, invite-only sign-up (POST /auth/signup) and Google
// (GET /auth/google, then POST /auth/google/exchange with the callback's one-time code). The
// session token and its expiry are kept in localStorage; every brain call sends authHeaders().
import type { AuthOptions, GoogleExchangeRequest, LoginRequest, LoginResponse, SignupRequest } from "@moe/contracts";
import { apiUrl } from "./apiUrl";

const TOKEN_KEY = "session.token";
const EXPIRES_KEY = "session.expires_at";
/** Fired on window when this tab signs in or out; other tabs see the storage event instead. */
export const SESSION_EVENT = "session-change";

/** A failed login. `status` is 401 for a wrong email or password, 429 after too many failures
 * (`retryAfter` seconds), 503 when the brain's sign-in is not configured. */
export class LoginError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly retryAfter: number | null = null,
  ) {
    super(message);
    this.name = "LoginError";
  }
}

/** localStorage, or null during server rendering or when the browser blocks it. */
function storage(): Storage | null {
  if (typeof window === "undefined") return null;
  try {
    return window.localStorage;
  } catch {
    return null;
  }
}

/** POSTs `body` to a brain route that answers with a session, and keeps the session. */
async function startSession(path: string, body: object, failure: string): Promise<LoginResponse> {
  const response = await fetch(apiUrl(path), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    const detail = await response
      .json()
      .then((b: { detail?: unknown }) => (typeof b.detail === "string" ? b.detail : null))
      .catch(() => null);
    const retryAfter = Number(response.headers.get("Retry-After")) || null;
    throw new LoginError(detail ?? `${failure} (${response.status})`, response.status, retryAfter);
  }
  const session = (await response.json()) as LoginResponse;
  const store = storage();
  store?.setItem(TOKEN_KEY, session.token);
  store?.setItem(EXPIRES_KEY, session.expires_at);
  notify();
  return session;
}

export function login(email: string, password: string): Promise<LoginResponse> {
  const body: LoginRequest = { email, password };
  return startSession("/auth/login", body, "Sign-in failed");
}

/** Invite-only (#128): 403 for an email nobody invited, 409 when it already has an account. */
export function signUp(name: string, email: string, password: string): Promise<LoginResponse> {
  const body: SignupRequest = { name, email, password };
  return startSession("/auth/signup", body, "Creating the account failed");
}

/** The one-time code the brain's Google callback put on /login?google=, for the session. */
export function finishGoogleSignIn(code: string): Promise<LoginResponse> {
  const body: GoogleExchangeRequest = { code };
  return startSession("/auth/google/exchange", body, "Google sign-in failed");
}

/** Where the Google button goes: the brain starts the sign-in and sends the browser to Google. */
export function googleSignInUrl(next: string): string {
  return apiUrl(`/auth/google?${new URLSearchParams({ next })}`);
}

/** What the sign-in page can offer; nothing extra when the brain can't be asked. */
export async function authOptions(): Promise<AuthOptions> {
  try {
    const response = await fetch(apiUrl("/auth/options"));
    if (response.ok) return (await response.json()) as AuthOptions;
  } catch {
    // unreachable brain: the page still offers email sign-in, which reports the problem
  }
  return { signup: false, google: false };
}

/** The current session token, or null when there is none or it has expired (and is dropped). */
export function sessionToken(): string | null {
  const store = storage();
  const token = store?.getItem(TOKEN_KEY);
  const expires = Date.parse(store?.getItem(EXPIRES_KEY) ?? "");
  if (!token || !(expires > Date.now())) {
    signOut();
    return null;
  }
  return token;
}

export function signOut(): void {
  const store = storage();
  if (!store?.getItem(TOKEN_KEY) && !store?.getItem(EXPIRES_KEY)) return;
  store.removeItem(TOKEN_KEY);
  store.removeItem(EXPIRES_KEY);
  notify();
}

function notify(): void {
  if (typeof window !== "undefined") window.dispatchEvent(new Event(SESSION_EVENT));
}

/** Calls `listener` whenever the session may have changed, in this tab or another. */
export function onSessionChange(listener: () => void): () => void {
  const onStorage = (e: StorageEvent) => {
    if (e.key === null || e.key === TOKEN_KEY || e.key === EXPIRES_KEY) listener();
  };
  window.addEventListener(SESSION_EVENT, listener);
  window.addEventListener("storage", onStorage);
  return () => {
    window.removeEventListener(SESSION_EVENT, listener);
    window.removeEventListener("storage", onStorage);
  };
}

/** `Authorization: Bearer <token>` for brain calls; empty when signed out. */
export function authHeaders(): Record<string, string> {
  const token = sessionToken();
  return token ? { Authorization: `Bearer ${token}` } : {};
}
