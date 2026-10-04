import type { Source } from "@moe/contracts";
import Link from "next/link";
import { fmtT } from "@/lib/format";
import { Icon, type IconName } from "./Icon";

const ICON: Record<Source["kind"], IconName> = {
  meeting: "file-text",
  github_issue: "github",
  github_pr: "github",
  github_code: "github",
  github_release: "github",
  jira_issue: "square-check",
};

/** Evidence chips under an answer. `newTab` keeps a live meeting open behind the link. */
export function Sources({ sources, newTab = false }: { sources: Source[]; newTab?: boolean }) {
  if (!sources.length) return null;
  return (
    <div className="sources">
      {sources.map((s, i) => {
        const body = (
          <>
            <Icon name={ICON[s.kind]} />
            {s.label}
            {s.kind === "meeting" && s.t != null ? ` · ${fmtT(s.t)}` : ""}
          </>
        );
        const key = `${s.kind}:${s.label}:${i}`;
        if (s.meeting_id) {
          return (
            <Link key={key} className="source" href={`/meetings/${s.meeting_id}`} target={newTab ? "_blank" : undefined}>
              {body}
            </Link>
          );
        }
        if (s.url) {
          return (
            <a key={key} className="source" href={s.url} target="_blank" rel="noopener noreferrer">
              {body}
            </a>
          );
        }
        return (
          <span key={key} className="source">
            {body}
          </span>
        );
      })}
    </div>
  );
}

/** Sources the answer could not reach. Shown as they are, never papered over. */
export function Unavailable({ items }: { items: string[] }) {
  if (!items.length) return null;
  return <p className="unavailable">Couldn&apos;t reach: {items.join(", ")}.</p>;
}
