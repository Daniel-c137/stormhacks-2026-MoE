-- A GitHub token is now connected per repository: each row is one repository of the team (its
-- owner/name in lower case) and the fine-grained token it is read with, through GitHub's hosted
-- MCP server. A repository without a row is read from GITHUB_MCP_URL with no credentials.
-- Tokens connected for a whole team before have no repository and are dropped: an admin
-- connects each repository with its token again.
delete from github_accounts;
alter table github_accounts drop constraint github_accounts_pkey;
alter table github_accounts add column repo text not null;
alter table github_accounts add primary key (team_id, repo);
