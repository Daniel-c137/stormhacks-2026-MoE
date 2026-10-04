-- The Atlassian account an admin connected for a team: approved task drafts are created as
-- issues on its Jira Cloud site with it. The API token is stored encrypted (brain.sealing, keyed
-- from AUTH_SECRET) and is never sent to a browser.
create table jira_accounts (
    team_id text primary key references teams on delete cascade,
    site text not null,
    email text not null,
    sealed_token text not null,
    connected_by text not null,
    connected_at timestamptz not null default now()
);

alter table jira_accounts enable row level security;
