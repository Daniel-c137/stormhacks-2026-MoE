-- Personal agendas: everyone in a meeting has their own agenda, which nobody else sees or edits,
-- and Polaris keeps time against each one. An agenda is keyed by meeting and person; one saved
-- before this belonged to the whole meeting, so it becomes the host's.
alter table agendas add column person_id text;
update agendas a set person_id = m.host_id from meetings m where m.id = a.meeting_id;
alter table agendas alter column person_id set not null;

alter table agenda_items add column person_id text;
update agenda_items i set person_id = a.person_id from agendas a where a.meeting_id = i.meeting_id;
alter table agenda_items alter column person_id set not null;

alter table agenda_items drop constraint agenda_items_meeting_id_fkey;
alter table agenda_items drop constraint agenda_items_pkey;
alter table agendas drop constraint agendas_pkey;
alter table agendas add primary key (meeting_id, person_id);
alter table agenda_items add primary key (meeting_id, person_id, ord);
alter table agenda_items add constraint agenda_items_agenda_fkey
    foreign key (meeting_id, person_id) references agendas (meeting_id, person_id) on delete cascade;
