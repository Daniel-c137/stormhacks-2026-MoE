// What the Settings voice select shows. No imports, so `pnpm --filter board test` runs it with Node alone.

/** The parts of a contracts `Voice` the select uses. */
export interface VoiceOption {
  id: string;
  name: string;
  sample: string;
  default_label?: string | null;
}

export interface VoiceChoice<V extends VoiceOption> {
  /** The select's value: a saved voice id, or "" for no saved voice (the agent's default). */
  value: string;
  /** The agent's default voice when the account lists it (the brain labels it). */
  defaultVoice: V | undefined;
  /** A saved voice the account no longer lists. */
  unavailable: boolean;
  /** The voice Speak and Listen use, so the one Preview plays; undefined when it isn't listed. */
  shown: V | undefined;
}

/** No saved voice (null, undefined or "") is the agent's default, not the first voice listed:
 * Speak and Listen fall back to the configured default voice. */
export function voiceChoice<V extends VoiceOption>(voices: V[], saved: string | null | undefined): VoiceChoice<V> {
  const defaultVoice = voices.find((v) => v.default_label);
  if (!saved) return { value: "", defaultVoice, unavailable: false, shown: defaultVoice };
  const listed = voices.find((v) => v.id === saved);
  return { value: saved, defaultVoice, unavailable: listed === undefined, shown: listed };
}

/** The option for no saved voice, named after the default voice when the list has it. */
export function defaultOptionLabel(defaultVoice: VoiceOption | undefined): string {
  return defaultVoice ? `Default (${defaultVoice.name})` : "Default";
}

/** The select's value as TeamSettings.voice: "" (Default) saves null. */
export function savedVoice(value: string): string | null {
  return value || null;
}

/** A voice's sample is either a clip to play or the line it would say. */
export function isClip(sample: string): boolean {
  return /^https?:\/\//.test(sample);
}
