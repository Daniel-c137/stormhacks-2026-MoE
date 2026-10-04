// node --test: `pnpm --filter board test`
import assert from "node:assert/strict";
import { test } from "node:test";
import { defaultOptionLabel, isClip, savedVoice, voiceChoice } from "./voiceChoice.ts";

const aria = { id: "aria", name: "Aria", sample: "https://x/aria.mp3", default_label: "Polaris's default voice" };
const roger = { id: "roger", name: "Roger", sample: "https://x/roger.mp3" };
const sarah = { id: "sarah", name: "Sarah", sample: "" };

test("no saved voice shows Default and previews the labelled default", () => {
  for (const saved of [null, undefined, ""]) {
    const choice = voiceChoice([aria, roger], saved);
    assert.equal(choice.value, "");
    assert.equal(choice.shown, aria);
    assert.equal(choice.unavailable, false);
    assert.equal(defaultOptionLabel(choice.defaultVoice), "Default (Aria)");
  }
});

test("no saved voice and no labelled default is Default, not the first voice", () => {
  const choice = voiceChoice([roger, sarah], null);
  assert.equal(choice.value, "");
  assert.equal(choice.shown, undefined);
  assert.equal(defaultOptionLabel(choice.defaultVoice), "Default");
});

test("a saved voice is shown, the first one and the default one included", () => {
  assert.deepEqual(voiceChoice([roger, sarah], "roger"), { value: "roger", defaultVoice: undefined, unavailable: false, shown: roger });
  const choice = voiceChoice([aria, roger], "aria");
  assert.equal(choice.value, "aria");
  assert.equal(choice.shown, aria);
});

test("a saved voice the account no longer lists stays selected, marked unavailable", () => {
  const choice = voiceChoice([aria, roger], "gone123");
  assert.equal(choice.value, "gone123");
  assert.equal(choice.unavailable, true);
  assert.equal(choice.shown, undefined);
});

test("Default saves null; a voice saves its id", () => {
  assert.equal(savedVoice(""), null);
  assert.equal(savedVoice("roger"), "roger");
});

test("a sample is a clip only when it is an http(s) link", () => {
  assert.equal(isClip("https://x/a.mp3"), true);
  assert.equal(isClip("http://localhost/a.mp3"), true);
  assert.equal(isClip("Hello, I'm Aria."), false);
  assert.equal(isClip(""), false);
});
