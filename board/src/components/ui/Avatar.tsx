import type { Person } from "@moe/contracts";
import type { CSSProperties } from "react";
import { cx } from "@/lib/format";

// Colour by initials, so each person keeps one colour everywhere.
const COLORS = ["#4F7DFF", "#0FA295", "#D9468F", "#D98E1C", "#7C5CFA", "#23A05E", "#2F95D3", "#8A6247", "#5B66E0"];

export function avatarColor(initials: string): string {
  const code = initials.toUpperCase().charCodeAt(0);
  return Number.isNaN(code) || code < 65 ? "#7A8190" : COLORS[(code - 65) % COLORS.length];
}

export interface AvatarProps {
  person: Pick<Person, "name" | "initials" | "photo_url">;
  size?: "sm" | "lg";
  /** The signed-in user, drawn in the accent colour when they have no photo. */
  me?: boolean;
  title?: string;
}

export function Avatar({ person, size, me, title }: AvatarProps) {
  const style: CSSProperties = person.photo_url
    ? { backgroundImage: `url("${person.photo_url}")`, backgroundSize: "cover", backgroundPosition: "center" }
    : me
      ? {}
      : { background: avatarColor(person.initials) };
  return (
    <span
      className={cx("av", size, me && !person.photo_url && "av-me")}
      style={style}
      title={title}
      role={person.photo_url ? "img" : undefined}
      aria-label={person.photo_url ? person.name : undefined}
      aria-hidden={person.photo_url ? undefined : true}
    >
      {person.photo_url ? null : person.initials}
    </span>
  );
}
