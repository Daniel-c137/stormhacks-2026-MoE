export type MarkState = "idle" | "capturing" | "working" | "speaking" | "hand";

const ANIMATION: Record<MarkState, string> = {
  idle: "none",
  capturing: "pTwinkle 1.4s ease-in-out infinite",
  working: "spin 3.2s linear infinite",
  speaking: "pTwinkle .7s ease-in-out infinite",
  hand: "none",
};

/** The north-star mark. `tile` puts it on the dark rounded square used as the agent's avatar. */
export function Mark({ size, state = "idle", tile = false }: { size: number; state?: MarkState; tile?: boolean }) {
  const raised = state === "hand";
  const ink = tile ? "var(--brand-star)" : "currentColor";
  return (
    <svg
      viewBox="-50 -50 100 100"
      width={size}
      height={size}
      aria-hidden="true"
      style={{ display: "block", overflow: "visible", width: size, height: size }}
    >
      {tile && <rect x={-50} y={-50} width={100} height={100} rx={22.6} style={{ fill: "var(--brand-tile)" }} />}
      <g transform={tile ? "scale(.72)" : undefined}>
        {raised && (
          <circle
            r={22}
            style={{
              fill: "none",
              stroke: "var(--coral)",
              strokeWidth: 2,
              transformBox: "fill-box",
              transformOrigin: "center",
              animation: "aHalo 1.6s ease-out infinite",
            }}
          />
        )}
        <g style={{ transformBox: "fill-box", transformOrigin: "center", animation: ANIMATION[state] }}>
          <path
            d="M0 -46 Q5 -5 40 0 Q5 5 0 46 Q-5 5 -40 0 Q-5 -5 0 -46Z"
            style={{
              fill: raised ? "var(--coral)" : "none",
              stroke: raised ? "var(--coral)" : ink,
              strokeWidth: tile ? 4.2 : 3.4,
              strokeLinejoin: "round",
              transition: "fill .3s, stroke .3s",
            }}
          />
          <circle r={raised ? 5 : 3.6} style={{ fill: raised ? (tile ? "var(--brand-tile)" : "var(--bg)") : ink }} />
        </g>
      </g>
    </svg>
  );
}
