import type { TaskDraft, TaskPushRequest } from "@moe/contracts";

export interface TaskReviewProps {
  tasks: TaskDraft[];
  onChange: (task: TaskDraft) => void;
  /** Explicit human approval of the chosen drafts and destination. */
  onPush: (request: TaskPushRequest) => void;
}

/** Edit title, owner, due and description; include or exclude; then push. */
export function TaskReview(_props: TaskReviewProps) {
  return null;
}
