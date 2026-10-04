-- Who marked an agenda item covered and when, so the room can tell the agent's call from a
-- person's check. covered_by is a person id or the agent's participant id; covered_t is seconds
-- from the meeting start, null when the meeting had not started.
alter table agenda_items
    add column covered_by text,
    add column covered_t double precision;
