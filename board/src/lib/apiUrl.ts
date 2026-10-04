// Where the brain is. NEXT_PUBLIC_API_URL may be absolute (http://localhost:8000) or relative to
// the board's own origin (/api, as in production); either way paths are appended to it as they are.
const BASE = (process.env.NEXT_PUBLIC_API_URL ?? "").trim().replace(/\/+$/, "");

/** False when NEXT_PUBLIC_API_URL is unset; nothing can reach the brain then. */
export const apiConfigured = (): boolean => BASE !== "";

/** Where the brain is, for messages. */
export const apiBase = (): string => BASE;

/** `path` (starting with "/") under the brain's URL. */
export function apiUrl(path: string): string {
  if (!BASE) throw new Error("NEXT_PUBLIC_API_URL is not set.");
  return `${BASE}${path.startsWith("/") ? path : `/${path}`}`;
}
