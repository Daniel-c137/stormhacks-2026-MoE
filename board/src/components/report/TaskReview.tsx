import type { Person, TaskDestination, TaskDraft } from "@moe/contracts";
import { Icon, Spinner } from "@/components/ui/Icon";
import { fmtT } from "@/lib/format";

export interface TaskReviewProps {
  tasks: TaskDraft[];
  members: Person[];
  /** Pushed or mid-push: the drafts can no longer be edited. */
  locked: boolean;
  pushing: boolean;
  pushed: boolean;
  destination: TaskDestination;
  /** What the destination is called, e.g. the Jira project key. */
  destinationLabel: string;
  /** Link for a pushed task's key, when the destination's address is known. */
  keyUrl: (task: TaskDraft) => string | null;
  status: string;
  onChange: (task: TaskDraft) => void;
  /** Persist a draft once the person has finished editing a field. */
  onCommit: (task: TaskDraft) => void;
  onDestination: (destination: TaskDestination) => void;
  /** Explicit human approval of the chosen drafts and destination. */
  onPush: () => void;
  /** Show the moment a task came from in the transcript; null when there is no transcript. */
  onJump: ((t: number) => void) | null;
  /** Only the meeting's host or an admin approves a push; everyone may edit the drafts. */
  canPush: boolean;
}

/** Edit title, owner, due and description; include or exclude; then push. */
export function TaskReview(props: TaskReviewProps) {
  const { tasks, members, locked, pushing, pushed, destination, destinationLabel } = props;
  const included = tasks.filter((t) => t.include).length;

  return (
    <>
      <div className="tasks">
        {tasks.length === 0 && <p className="muted-p">No tasks came out of this meeting.</p>}
        {tasks.map((task) => {
          // Fields save when they lose focus; pickers and the checkbox save at once.
          const edit = (patch: Partial<TaskDraft>) => props.onChange({ ...task, ...patch });
          const editNow = (patch: Partial<TaskDraft>) => {
            const next = { ...task, ...patch };
            props.onChange(next);
            props.onCommit(next);
          };
          const url = task.key ? props.keyUrl(task) : null;
          return (
            <div key={task.id} className="task" data-inc={task.include}>
              <input
                type="checkbox"
                checked={task.include}
                onChange={(e) => editNow({ include: e.target.checked })}
                disabled={locked}
                aria-label={`Include “${task.title}”`}
              />
              <div className="task-fields">
                <div className="task-head full">
                  <input
                    className="field t-title"
                    value={task.title}
                    onChange={(e) => edit({ title: e.target.value })}
                    onBlur={() => props.onCommit(task)}
                    disabled={locked}
                    aria-label="Task title"
                  />
                  {task.t != null && (
                    <button
                      type="button"
                      className="jump"
                      onClick={() => task.t != null && props.onJump?.(task.t)}
                      disabled={!props.onJump}
                      aria-label={`Jump to ${fmtT(task.t)} in the transcript`}
                    >
                      <Icon name="scroll-text" />
                      {fmtT(task.t)}
                    </button>
                  )}
                </div>
                <label className="label">
                  Description
                  <textarea
                    className="field"
                    value={task.description ?? ""}
                    onChange={(e) => edit({ description: e.target.value || null })}
                    onBlur={() => props.onCommit(task)}
                    disabled={locked}
                    rows={1}
                  />
                </label>
                <label className="label">
                  Owner
                  {/* Empty stays empty: an owner is only set when the meeting named one. */}
                  <select
                    className="field"
                    value={task.owner_id ?? ""}
                    onChange={(e) => editNow({ owner_id: e.target.value || null })}
                    disabled={locked}
                  >
                    <option value="">No owner</option>
                    {task.owner_id && !members.some((m) => m.id === task.owner_id) && (
                      <option value={task.owner_id}>Unknown member</option>
                    )}
                    {members.map((m) => (
                      <option key={m.id} value={m.id}>
                        {m.name}
                      </option>
                    ))}
                  </select>
                </label>
                <label className="label">
                  Due
                  <input
                    className="field"
                    type="date"
                    value={task.due ?? ""}
                    onChange={(e) => editNow({ due: e.target.value || null })}
                    disabled={locked}
                  />
                </label>
                <div className="task-src">
                  {task.key &&
                    (url ? (
                      <a className="key" href={url} target="_blank" rel="noopener noreferrer">
                        <Icon name="circle-check" />
                        {task.key}
                      </a>
                    ) : (
                      <span className="key">
                        <Icon name="circle-check" />
                        {task.key}
                      </span>
                    ))}
                </div>
              </div>
            </div>
          );
        })}
      </div>
      {tasks.length > 0 && props.canPush && (
        <div className="push-row">
          <button
            type="button"
            className={pushed ? "btn btn-primary pushed" : "btn btn-primary"}
            onClick={props.onPush}
            disabled={pushed || pushing || included === 0}
          >
            {pushing ? <Spinner size={16} /> : <Icon name={pushed ? "circle-check" : "upload"} />}
            {pushing
              ? `Pushing to ${destinationLabel}…`
              : pushed
                ? `Pushed to ${destinationLabel}`
                : `Push ${included} task${included === 1 ? "" : "s"} to ${destinationLabel}`}
          </button>
          {!locked && (
            <select
              className="field"
              value={destination}
              onChange={(e) => props.onDestination(e.target.value as TaskDestination)}
              aria-label="Where to push the tasks"
            >
              <option value="jira">Jira</option>
              <option value="github">GitHub</option>
            </select>
          )}
          <span className="push-st" role="status">
            {props.status}
          </span>
        </div>
      )}
    </>
  );
}
