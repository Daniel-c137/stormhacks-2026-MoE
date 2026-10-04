"use client";

import { type ResponseActionName, type ResponseCard, identity } from "@moe/contracts";
import { useCallback, useRef, useState } from "react";
import { Icon } from "@/components/ui/Icon";
import { Sources, Unavailable } from "@/components/ui/Sources";
import { useDismiss } from "@/hooks/useDismiss";

export interface ResponseCardViewProps {
  card: ResponseCard;
  /** From TeamSettings.who_can_allow; everyone by default. */
  canAct: boolean;
  /** An action was sent and the agent has not confirmed it yet. */
  busy: boolean;
  onAction: (action: ResponseActionName) => void;
}

/** Shared answer with sources. Silent until Speak; also Send to public chat, Dismiss, Show on stage. */
export function ResponseCardView({ card, canAct, busy, onAction }: ResponseCardViewProps) {
  const agent = identity.agent_name;
  const [open, setOpen] = useState(false);
  const wrap = useRef<HTMLDivElement>(null);
  useDismiss(
    wrap,
    open,
    useCallback(() => setOpen(false), []),
  );
  const sayLabel = canAct ? "Say it out loud" : `Only the host can let ${agent} speak`;
  const { invocation, answer } = card;

  return (
    <div className="atlas-wrap" ref={wrap}>
      <div className="ready" role="group" aria-label={`${agent} has an answer ready`}>
        <Icon name="square-check" className="lead" />
        <button type="button" className="ready-label" onClick={() => setOpen((o) => !o)} aria-expanded={open} aria-haspopup="dialog">
          Answer ready
          <Icon name={open ? "chevron-up" : "chevron-down"} />
        </button>
        <span className="sep" aria-hidden="true" />
        <div className="ready-act has-tip">
          <button type="button" disabled={busy} onClick={() => onAction("send_to_chat")} aria-label="Post in chat">
            <Icon name="message-square-text" />
          </button>
          <span role="tooltip" className="tip">
            Post in chat
          </span>
        </div>
        <div className="ready-act has-tip">
          <button type="button" disabled={!canAct || busy} onClick={() => onAction("speak")} aria-label={sayLabel}>
            <Icon name="volume-2" />
          </button>
          <span role="tooltip" className="tip">
            {sayLabel}
          </span>
        </div>
      </div>
      {open && (
        <div className="card-pop" role="dialog" aria-label={`${agent}'s answer`}>
          <p className="card-q">
            <b>{invocation.asked_by_name} asked</b> “{invocation.question}”
          </p>
          <p className="card-a">{answer.text}</p>
          <Sources sources={answer.sources} newTab />
          <Unavailable items={answer.unavailable} />
          <div className="card-acts">
            {answer.snippets.length > 0 && (
              <button type="button" className="btn btn-outline btn-sm" onClick={() => onAction("show_on_stage")}>
                <Icon name="file-code" />
                Show the code
              </button>
            )}
            <button type="button" className="btn btn-quiet btn-sm" disabled={busy} onClick={() => onAction("dismiss")}>
              <Icon name="x" />
              Dismiss
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
