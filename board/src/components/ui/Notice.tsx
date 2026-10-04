import type { ReactNode } from "react";
import { describeError } from "@/lib/api";
import { Icon } from "./Icon";

/** Says plainly that something is missing or failed, with a way to try again. */
export function Notice({ error, children, onRetry }: { error?: unknown; children?: ReactNode; onRetry?: () => void }) {
  return (
    <p className="notice" role="status">
      <Icon name="info" />
      <span>
        {children}
        {children && error ? " " : ""}
        {error ? describeError(error) : ""}
        {onRetry && (
          <button type="button" className="link" onClick={onRetry}>
            Try again
          </button>
        )}
      </span>
    </p>
  );
}
