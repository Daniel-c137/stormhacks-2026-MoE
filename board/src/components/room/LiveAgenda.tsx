import type { Agenda, AgendaItem } from "@moe/contracts";
import { Icon, type IconName } from "@/components/ui/Icon";
import { Notice } from "@/components/ui/Notice";

type State = "current" | AgendaItem["status"];

const ICON: Record<State, IconName> = {
  current: "circle-dot",
  pending: "circle",
  covered: "circle-check",
  skipped: "circle-dashed",
};

const STATE_LABEL: Record<State, string> = {
  current: "Now",
  pending: "Not yet",
  covered: "Covered",
  skipped: "Skipped",
};

/** "3 min", "45 s": talk time so far, rounded down so it never runs ahead of the tracker. */
function used(seconds: number): string {
  return seconds < 60 ? `${Math.floor(seconds)} s` : `${Math.floor(seconds / 60)} min`;
}

function Time({ item }: { item: AgendaItem }) {
  const spent = item.discussed_s ?? 0;
  if (!item.minutes) return spent > 0 ? <span className="la-time">{used(spent)}</span> : null;
  const over = spent - item.minutes * 60;
  return (
    <span className="la-time" data-over={over > 0}>
      {over > 0 ? `${used(over)} over ${item.minutes} min` : `${used(spent)} / ${item.minutes} min`}
    </span>
  );
}

/** The live agenda in the side panel: what is being discussed now, time used against each
 * timebox, and what is covered or skipped. Times come from the agent's tracker, as they arrive. */
export function LiveAgenda({ agenda, error }: { agenda: Agenda | null; error: Error | null }) {
  if (!agenda && error) return <Notice error={error}>The agenda couldn&apos;t be loaded.</Notice>;
  if (!agenda) return <p className="msgs-empty">Loading the agenda…</p>;
  if (!agenda.items.length) return <p className="msgs-empty">This meeting has no agenda.</p>;
  const current = agenda.items.find((i) => i.id === agenda.current_item_id) ?? null;
  return (
    <div className="la">
      <p className="la-now" aria-live="polite">
        {current ? (
          <>
            Now: <b>{current.title}</b>
          </>
        ) : (
          "Not on an agenda item right now"
        )}
      </p>
      <ol className="la-list">
        {agenda.items.map((item) => {
          const state: State = item.id === agenda.current_item_id ? "current" : item.status;
          const spent = item.discussed_s ?? 0;
          const share = item.minutes ? Math.min(spent / (item.minutes * 60), 1) : 0;
          const over = Boolean(item.minutes && spent > item.minutes * 60);
          return (
            <li key={item.id} className="la-item" data-state={state} aria-current={state === "current" ? "step" : undefined}>
              <Icon name={ICON[state]} className="la-ic" />
              <div className="la-main">
                <span className="la-title">{item.title}</span>
                <span className="la-meta">
                  <span className="sr">{STATE_LABEL[state]}. </span>
                  <Time item={item} />
                </span>
                {item.minutes && state === "current" ? (
                  <span className="la-bar" data-over={over} aria-hidden="true">
                    <span style={{ width: `${share * 100}%` }} />
                  </span>
                ) : null}
              </div>
            </li>
          );
        })}
      </ol>
    </div>
  );
}
