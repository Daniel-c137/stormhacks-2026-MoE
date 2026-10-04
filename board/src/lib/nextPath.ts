// Where to go after signing in. No imports, so `pnpm --filter board test` runs it with Node alone.

/** The ?next= value as a path on this site (`origin`), else home. It is resolved the way the
 * browser would, so `/\evil.com` and `/<TAB>/evil.com`, which browsers read as //evil.com, go
 * home rather than off-site. */
export function safeNextPath(next: string | null | undefined, origin: string): string {
  if (!next) return "/";
  let url: URL;
  try {
    url = new URL(next, origin);
  } catch {
    return "/";
  }
  return url.origin === origin ? `${url.pathname}${url.search}${url.hash}` : "/";
}
