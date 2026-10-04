-- When the agent first joined the meeting's room, set once by the worker
-- (POST /internal/meetings/{id}/agent-joined). Null means it never joined, so the report and the
-- board do not list it as present.
alter table meetings add column agent_joined_at timestamptz;
