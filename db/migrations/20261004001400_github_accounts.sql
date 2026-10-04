-- The GitHub account whose fine-grained personal access token an admin connected for a team:
-- the team's GitHub reads (answers, fact checks, code search, connector status) go to GitHub's
-- hosted MCP server with it. A team without a row reads GITHUB_MCP_URL with no credentials.
-- The token is stored encrypted (brain.sealing, keyed from AUTH_SECRET and the team id) and is
-- never sent to a browser; `login` is its owner on GitHub, shown as @login.
create table github_accounts (
    team_id text primary key references teams on delete cascade,
    login text not null,
    sealed_token text not null,
    connected_by text not null,
    connected_at timestamptz not null default now()
);

alter table github_accounts enable row level security;
