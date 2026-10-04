-- Timekeeping against the agenda during a live meeting (#10). Times are seconds from the meeting
-- start, like transcript segments.
alter table agendas
    add column current_item_id text,  -- being discussed now; null when off the agenda
    add column tracked_until double precision;  -- transcript tracked so far; null before any

alter table agenda_items
    add column discussed_s double precision not null default 0 check (discussed_s >= 0),
    add column nudged_t double precision;  -- when the agent nudged that it had not come up
