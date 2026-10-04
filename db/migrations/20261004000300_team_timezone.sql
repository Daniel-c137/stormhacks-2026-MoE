-- The team's time zone, an IANA name such as America/Vancouver. Timestamps stay in UTC; dates
-- that people and prompts see (meeting days, "today", due dates) are read in this zone.
alter table team_settings add column timezone text not null default 'UTC';
