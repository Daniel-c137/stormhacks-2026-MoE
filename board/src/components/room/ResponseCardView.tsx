import type { ResponseActionName, ResponseCard } from "@moe/contracts";

export interface ResponseCardViewProps {
  card: ResponseCard;
  /** From TeamSettings.who_can_allow; everyone by default. */
  canAct: boolean;
  onAction: (action: ResponseActionName) => void;
}

/** Shared answer with sources. Silent until Speak; also Send to public chat, Dismiss, Show on stage. */
export function ResponseCardView(_props: ResponseCardViewProps) {
  return null;
}
