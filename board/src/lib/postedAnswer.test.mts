// node --test: `pnpm --filter board test`
import assert from "node:assert/strict";
import { test } from "node:test";
import { postedAnswerText } from "./postedAnswer.ts";

const ANSWER = "The fix is PR #50, merged Oct 2. The latest release is v0.9.3, so it is not released yet.";

test("an answer posted with its sources and unavailable lines is the answer's own text", () => {
  const posted = `${ANSWER}\n\nSources: dropsubs/dropsubs#50 (https://github.com/dropsubs/dropsubs/pull/50), dropsubs/dropsubs@v0.9.3\n\nUnavailable: Jira is not configured`;
  assert.equal(postedAnswerText(posted), ANSWER);
});

test("an answer posted without sources is unchanged", () => {
  assert.equal(postedAnswerText(ANSWER), ANSWER);
});

test("only the trailing lines the agent adds are cut, not the word Sources in the answer", () => {
  const text = "Sources: the team keeps a list.\n\nSources: dropsubs/dropsubs#7";
  assert.equal(postedAnswerText(text), "Sources: the team keeps a list.");
});
