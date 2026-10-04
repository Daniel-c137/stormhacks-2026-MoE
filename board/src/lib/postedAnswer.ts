// The agent posts an answer to chat as its text, then "Sources: …" and "Unavailable: …" as
// paragraphs of their own (the worker's chat_text). Matched to its card, the answer shows its
// sources as links instead, so those lines are cut.
const TRAILER = /\n\n(?:Sources|Unavailable): [^\n]*$/;

/** The answer's own text in a message the agent posted to chat. */
export function postedAnswerText(text: string): string {
  let body = text;
  while (TRAILER.test(body)) body = body.replace(TRAILER, "");
  return body;
}
