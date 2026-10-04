-- Admins: the only accounts that change team settings and create accounts, and that may also do
-- what a meeting's host does. Read from the store on every request, never from the session
-- token, so a revoke takes effect at once.
alter table people add column is_admin boolean not null default false;
