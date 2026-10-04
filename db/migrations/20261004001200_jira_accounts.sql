-- The Atlassian account an admin connected for a team: approved task drafts are created as
-- issues in `project` on its Jira Cloud site with it. Its own site and project, apart from the
-- Jira project the agent reads (team_settings.jira). The API token is stored encrypted
-- (brain.sealing, keyed from AUTH_SECRET and the team id) and is never sent to a browser.
-- issue_type_id is the project's issue type tasks are created as, chosen when connecting.
create table jira_accounts (
    team_id text primary key references teams on delete cascade,
    site text not null,
    project text not null,
    issue_type_id text,
    email text not null,
    sealed_token text not null,
    connected_by text not null,
    connected_at timestamptz not null default now()
);

alter table jira_accounts enable row level security;

-- Where a pushed task's issue opens: its site can differ from the one the agent reads.
alter table task_drafts add column url text;
