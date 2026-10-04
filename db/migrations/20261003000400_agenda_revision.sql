-- Every agenda save bumps the revision. Timekeeping ticks and lobby edits save only if the
-- agenda is still at the revision they read, so neither silently overwrites the other (#10).
alter table agendas add column revision integer not null default 0 check (revision >= 0);

-- Agendas saved before this migration count as saved once; 0 means none saved yet.
update agendas set revision = 1;
