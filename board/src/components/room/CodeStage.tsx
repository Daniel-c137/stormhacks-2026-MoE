import type { CodeSnippet } from "@moe/contracts";

export interface CodeStageProps {
  snippet: CodeSnippet;
  onClose: () => void;
}

/** A snippet shown to everyone, with its GitHub link and highlighted lines. */
export function CodeStage(_props: CodeStageProps) {
  return null;
}
