// node --test: `pnpm --filter board test`
import assert from "node:assert/strict";
import { test } from "node:test";
import { defaultOptionLabel, isClip, savedVoice, voiceChoice, voiceOptions } from "./voiceChoice.ts";

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

test("a saved voice is shown, the first one included", () => {
  assert.deepEqual(voiceChoice([roger, sarah], "roger"), { value: "roger", defaultVoice: undefined, unavailable: false, shown: roger });
});

test("a team that saved the default voice's id sees Default, not a second copy of it", () => {
  const choice = voiceChoice([aria, roger], "aria");
  assert.equal(choice.value, "");
  assert.equal(choice.shown, aria);
  assert.equal(choice.unavailable, false);
});

test("the default voice is listed once, as Default, and the rest are sorted by name", () => {
  const zed = { id: "z", name: "zed", sample: "" };
  const v10 = { id: "v10", name: "Voice 10", sample: "" };
  const v9 = { id: "v9", name: "Voice 9", sample: "" };
  const options = voiceOptions([v10, roger, aria, zed, v9, sarah]);
  assert.deepEqual(
    options.map((v) => v.name),
    ["Roger", "Sarah", "Voice 9", "Voice 10", "zed"],
  );
});

test("with no labelled default every voice is listed", () => {
  assert.deepEqual(voiceOptions([sarah, roger]).map((v) => v.id), ["roger", "sarah"]);
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
