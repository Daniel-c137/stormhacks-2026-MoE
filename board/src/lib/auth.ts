// Sign-in against the brain (POST /auth/login). The session token and its expiry are kept in
// localStorage; every brain call sends authHeaders().
import type { LoginRequest, LoginResponse } from "@moe/contracts";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "";
const TOKEN_KEY = "session.token";
const EXPIRES_KEY = "session.expires_at";

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

export async function login(email: string, password: string): Promise<LoginResponse> {
  const body: LoginRequest = { email, password };
  const response = await fetch(`${API_URL}/auth/login`, {
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
    throw new LoginError(detail ?? `Sign-in failed (${response.status})`, response.status, retryAfter);
  }
  const session = (await response.json()) as LoginResponse;
  const store = storage();
  store?.setItem(TOKEN_KEY, session.token);
  store?.setItem(EXPIRES_KEY, session.expires_at);
  return session;
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
  store?.removeItem(TOKEN_KEY);
  store?.removeItem(EXPIRES_KEY);
}

/** `Authorization: Bearer <token>` for brain calls; empty when signed out. */
export function authHeaders(): Record<string, string> {
  const token = sessionToken();
  return token ? { Authorization: `Bearer ${token}` } : {};
}
