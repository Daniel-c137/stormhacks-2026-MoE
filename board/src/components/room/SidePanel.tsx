export type SidePanelTab = "agent" | "chat" | "transcript";

export interface SidePanelProps {
  tab: SidePanelTab;
  onTab: (tab: SidePanelTab) => void;
}

/** agent: fact-checks, code shown, questions answered. chat: public and private. transcript: live. */
export function SidePanel(_props: SidePanelProps) {
  return null;
}
