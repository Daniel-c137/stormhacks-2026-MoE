import type { TranscriptSegment } from "@moe/contracts";

const names = typeof Intl !== "undefined" && "DisplayNames" in Intl ? new Intl.DisplayNames(["en"], { type: "language" }) : null;

/** "es" -> "Spanish"; the code itself when the browser has no name for it. */
export function languageName(code: string): string {
  try {
    return names?.of(code) ?? code;
  } catch {
    return code;
  }
}

/**
 * How a segment's English came about (#106): translated, still being translated (a provisional
 * partial), or left untranslated because translation failed. Null for speech that was English.
 */
export function translationNote(seg: TranscriptSegment): string | null {
  const lang = seg.language;
  if (!lang || lang === "en") return null;
  const name = languageName(lang);
  if (seg.original_text) return seg.is_final ? `translated from ${name}` : `translating from ${name}…`;
  return seg.is_final ? `untranslated · ${name}` : null;
}
