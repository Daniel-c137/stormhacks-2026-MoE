// Where to go after signing in. No imports, so `pnpm --filter board test` runs it with Node alone.

/** The ?next= path when it is a safe same-site path, else home. */
export function safeNextPath(next: string | null | undefined, origin: string): string {
  void origin;
  return next && next.startsWith("/") && !next.startsWith("//") ? next : "/";
}
