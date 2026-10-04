-- Email and password sign-in, checked by the brain (POST /auth/login). Only the argon2 hash of a
-- password is kept. Accounts come from `brain add-user`; there is no public sign-up.
create table logins (
    person_id text primary key references people on delete cascade,
    email text not null,
    password_hash text not null,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);
-- One login per email, ignoring case; login_by_email looks it up the same way.
create unique index logins_email_idx on logins (lower(email));

alter table logins enable row level security;
