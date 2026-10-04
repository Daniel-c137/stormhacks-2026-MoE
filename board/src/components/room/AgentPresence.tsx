import { type AgentState, identity } from "@moe/contracts";

export interface AgentPresenceProps {
  state: AgentState;
  /** The Ask button was pressed and the agent has not confirmed it is listening yet. */
  askPending: boolean;
  /** False when the agent is not in the room; asking is then unavailable. */
  agentPresent: boolean;
  wake: string;
  onAsk: () => void;
  onCancel: () => void;
}

type View = "ask" | "listening" | "working" | "speaking";

const ROW = [
  [-7, 0],
  [0, 0],
  [7, 0],
];
const ORBIT = [
  [0, -5.5],
  [4.8, 2.8],
  [-4.8, 2.8],
];

/** Three dots: level bars while listening, an orbit while working, a talking pattern while speaking. */
function Glyph({ view }: { view: Exclude<View, "ask"> }) {
  const positions = view === "working" ? ORBIT : ROW;
  return (
    <span className="glyph" aria-hidden="true" style={{ animation: view === "working" ? "spin 1.15s linear infinite" : "none" }}>
      {positions.map(([dx, dy], i) => (
        <span
          key={i}
          style={{
            transform: `translate(calc(-50% + ${dx}px), calc(-50% + ${dy}px))`,
            animation:
              view === "listening"
                ? `atLevel .9s ease-in-out ${i * 0.15}s infinite`
                : view === "speaking"
                  ? `atTalk ${[0.7, 0.55, 0.8][i]}s ease-in-out ${[0, 0.2, 0.1][i]}s infinite`
                  : "none",
          }}
        />
      ))}
    </span>
  );
}

/** The agent button: Ask, Listening (press to cancel), Working, Speaking. A ready answer replaces it. */
export function AgentPresence({ state, askPending, agentPresent, wake, onAsk, onCancel }: AgentPresenceProps) {
  const agent = identity.agent_name;
  const view: View =
    state.state === "speaking"
      ? "speaking"
      : state.state === "working"
        ? "working"
        : state.state === "capturing" || askPending
          ? "listening"
          : "ask";
  const inert = view === "working" || view === "speaking" || (view === "ask" && !agentPresent);
  const label = { ask: `Ask ${agent}`, listening: "Listening", working: "Working", speaking: "Speaking" }[view];
  const aria = {
    ask: `Ask ${agent}. You can also say ${wake}.`,
    listening: `${agent} is listening for your question. Click to cancel.`,
    working: `${agent} is working on an answer`,
    speaking: `${agent} is speaking`,
  }[view];
  const tip =
    view === "ask"
      ? agentPresent
        ? `Or say “${wake}”`
        : `${agent} isn't in this meeting yet`
      : view === "listening"
        ? "Click to cancel"
        : state.detail;

  return (
    <div className="atlas-wrap has-tip">
      <button
        type="button"
        className="atlas-btn"
        data-state={view}
        onClick={inert ? undefined : view === "listening" ? onCancel : onAsk}
        aria-label={aria}
        aria-disabled={inert}
      >
        {view !== "ask" && <Glyph view={view} />}
        <span className="atlas-label">{label}</span>
      </button>
      {tip && (
        <span role="tooltip" className="tip">
          {tip}
        </span>
      )}
    </div>
  );
}
