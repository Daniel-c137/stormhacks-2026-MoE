"use client";

/**
 * Live translation of a meeting's non-English speech into English (#106). Off by default; the
 * host turns it on before anyone joins. Turning it on is the approval for sending that speech to
 * the AI model as it's spoken, so the line under it says so.
 */
export function TranslationSwitch({
  checked,
  onChange,
  disabled = false,
  note,
}: {
  checked: boolean;
  onChange?: (on: boolean) => void;
  disabled?: boolean;
  note?: string;
}) {
  return (
    <label className="check-row" data-readonly={disabled || !onChange}>
      <input
        type="checkbox"
        checked={checked}
        onChange={(e) => onChange?.(e.target.checked)}
        disabled={disabled || !onChange}
      />
      <span className="person-text">
        <span className="person-name">Translate other languages into English</span>
        <span className="person-meta">
          Non-English speech is sent to the AI model as it&apos;s spoken.{note ? ` ${note}` : ""}
        </span>
      </span>
    </label>
  );
}
