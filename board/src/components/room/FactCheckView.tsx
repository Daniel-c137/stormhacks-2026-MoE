"use client";

import { type FactCheck, type Verdict, identity } from "@moe/contracts";
import { useCallback, useRef, useState } from "react";
import { Icon, type IconName } from "@/components/ui/Icon";
import { Sources } from "@/components/ui/Sources";
import { useDismiss } from "@/hooks/useDismiss";
import { fmtClock, fmtT } from "@/lib/format";

const VERDICT: Record<Verdict, { label: string; icon: IconName }> = {
  contradicted: { label: "Contradicted", icon: "triangle-alert" },
  supported: { label: "Supported", icon: "circle-check" },
  unknown: { label: "Couldn't verify", icon: "circle-dashed" },
};

export function VerdictChip({ verdict }: { verdict: Verdict }) {
  const { label, icon } = VERDICT[verdict];
  return (
    <span className="verdict" data-verdict={verdict}>
      <Icon name={icon} />
      {label}
    </span>
  );
}

/** The claim, who said it, the verdict, how sure, and the evidence it rests on. */
export function FactCheckBody({ check, meId, showTime = false }: { check: FactCheck; meId: string; showTime?: boolean }) {
  const percent = Math.round(Math.min(Math.max(check.confidence, 0), 1) * 100);
  const mine = check.visibility === "private";
  return (
    <div className="fc-body">
      <div className="fc-head">
        <VerdictChip verdict={check.verdict} />
        {check.severity === "high" && <span className="fc-sev">High stakes</span>}
        {mine && (
          <span className="priv">
            <Icon name="lock" />
            {check.recipient_id === meId ? "Only you" : "Private"}
          </span>
        )}
        {showTime && check.created_at && <span className="msg-ts">{fmtClock(new Date(check.created_at))}</span>}
      </div>
      <p className="card-q">
        <b>{check.speaker_name} said</b>
        {check.t != null && <span className="fc-t"> at {fmtT(check.t)}</span>}
      </p>
      <p className="fc-claim">“{check.claim}”</p>
      <Sources sources={check.sources} newTab />
      {!check.sources.length && <p className="fc-none">No source was found for this claim.</p>}
      <p className="fc-conf">{percent}% confidence</p>
    </div>
  );
}

export interface FactCheckHandProps {
  check: FactCheck;
  meId: string;
  onDismiss: () => void;
}

/** The agent's raised hand for a high-stakes contradiction: a visual cue only, never spoken. Opens
 * the fact-check; each person dismisses it for themselves. */
export function FactCheckHand({ check, meId, onDismiss }: FactCheckHandProps) {
  const agent = identity.agent_name;
  const [open, setOpen] = useState(false);
  const wrap = useRef<HTMLDivElement>(null);
  useDismiss(
    wrap,
    open,
    useCallback(() => setOpen(false), []),
  );
  return (
    <div className="atlas-wrap" ref={wrap}>
      <button
        type="button"
        className="ready hand-up"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        aria-haspopup="dialog"
        aria-label={`${agent} raised a hand: ${check.speaker_name}'s claim may be wrong. Show the fact-check`}
      >
        <Icon name="hand" className="lead" />
        <span className="atlas-label">Fact-check</span>
        <Icon name={open ? "chevron-up" : "chevron-down"} />
      </button>
      {open && (
        <div className="card-pop" role="dialog" aria-label={`${agent}'s fact-check`}>
          <FactCheckBody check={check} meId={meId} />
          <div className="card-acts">
            <button type="button" className="btn btn-quiet btn-sm" onClick={onDismiss}>
              <Icon name="x" />
              Dismiss
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

/** A private fact-check, quietly, for the one person it is addressed to. */
export function PrivateCheckCard({ check, meId, onDismiss }: FactCheckHandProps) {
  const agent = identity.agent_name;
  return (
    <div className="fc-quiet" role="note" aria-label={`A private fact-check from ${agent}, only for you`}>
      <div className="fc-quiet-top">
        <span className="fc-quiet-from">{agent} · fact-check</span>
        <button type="button" className="icon-btn sm" onClick={onDismiss} aria-label="Dismiss this fact-check">
          <Icon name="x" />
        </button>
      </div>
      <FactCheckBody check={check} meId={meId} />
    </div>
  );
}
