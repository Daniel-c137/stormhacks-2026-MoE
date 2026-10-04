-- Admins: the only accounts that change the team's connectors and create accounts. Read from
-- the store on every request, never from the session token, so a revoke takes effect at once.
alter table people add column is_admin boolean not null default false;
