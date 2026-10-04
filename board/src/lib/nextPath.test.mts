// node --test: `pnpm --filter board test`
import assert from "node:assert/strict";
import { test } from "node:test";
import { safeNextPath } from "./nextPath.ts";

const ORIGIN = "https://skyroom.example";

test("a path on this site is kept with its query and fragment", () => {
  assert.equal(safeNextPath("/meetings/x?y=1#z", ORIGIN), "/meetings/x?y=1#z");
  assert.equal(safeNextPath("/", ORIGIN), "/");
  assert.equal(safeNextPath(`${ORIGIN}/m/abc`, ORIGIN), "/m/abc");
});

test("anything that would leave the site goes home instead", () => {
  for (const next of ["/\\evil.com", "/\t/evil.com", "/\n/evil.com", "//evil.com", "\\\\evil.com", "https://evil.com", "https://evil.com/login", "javascript:alert(1)", "http://skyroom.example/m"]) {
    assert.equal(safeNextPath(next, ORIGIN), "/", JSON.stringify(next));
  }
});

test("an encoded backslash stays a path on this site", () => {
  assert.equal(safeNextPath("/%5Cevil.com", ORIGIN), "/%5Cevil.com");
});

test("no next is home, and a bare word stays on this site", () => {
  for (const next of [null, undefined, ""]) assert.equal(safeNextPath(next, ORIGIN), "/", JSON.stringify(next));
  assert.equal(safeNextPath("evil.com", ORIGIN), "/evil.com");
});
