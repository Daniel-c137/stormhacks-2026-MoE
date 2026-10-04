// Links to the team's tools. No imports, so `pnpm --filter board test` runs it with Node alone.

/** A Jira issue's page. The site is stored as acme.atlassian.net, or with a scheme and a
 * trailing slash as sites were saved before; either gives https://acme.atlassian.net/browse/KEY. */
export function jiraIssueUrl(site: string | null | undefined, key: string | null | undefined): string | null {
  const host = (site ?? "").trim().replace(/^[a-z][a-z0-9+.-]*:\/\//i, "").replace(/\/+$/, "");
  return host && key ? `https://${host}/browse/${encodeURIComponent(key)}` : null;
}

export type CodeHost = "github" | "gitlab";

/** The code host of a link to lines of code: GitLab writes /-/blob/, GitHub /blob/. */
export function codeHost(url: string): CodeHost {
  try {
    const { hostname, pathname } = new URL(url);
    return hostname !== "github.com" && pathname.includes("/-/blob/") ? "gitlab" : "github";
  } catch {
    return "github";
  }
}

export const CODE_HOST_NAME: Record<CodeHost, string> = { github: "GitHub", gitlab: "GitLab" };
