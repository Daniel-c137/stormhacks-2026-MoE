import type { Meeting, Person } from "@moe/contracts";

/** The brain's rule for what a meeting's host does (end it, change its invitees, retry its
 * write-up): the host or an admin, so a meeting whose host left can still be managed. Pushing its
 * tasks to Jira is an admin's alone. */
export const hostOrAdmin = (meeting: Pick<Meeting, "host_id">, me: Person): boolean => meeting.host_id === me.id || me.is_admin === true;
