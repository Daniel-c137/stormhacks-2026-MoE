-- Several code repositories per team, and GitLab next to GitHub.
-- team_settings.github was one repository, {"repo", "ref", "connected", "indexed_at", "files"};
-- it becomes {"repos": [{"path", "ref", "connected", "indexed_at", "files"}]}, one entry for the
-- old repository (none when it was unset). gitlab holds GitLab projects the same way.
update team_settings
set github = jsonb_build_object(
    'repos',
    case
        when coalesce(btrim(github ->> 'repo'), '') = '' then '[]'::jsonb
        else jsonb_build_array(
            jsonb_build_object(
                'path', btrim(github ->> 'repo'),
                'ref', github -> 'ref',
                'connected', coalesce(github -> 'connected', 'false'::jsonb),
                'indexed_at', github -> 'indexed_at',
                'files', github -> 'files'
            )
        )
    end
)
where not github ? 'repos';

alter table team_settings add column gitlab jsonb not null default '{"projects": []}';

-- The board links Jira issues as https://<site>/browse/KEY, so the site is kept without a scheme
-- or trailing slash ("https://acme.atlassian.net/" becomes acme.atlassian.net).
update team_settings
set jira = jsonb_set(
    jira,
    '{site}',
    to_jsonb(rtrim(regexp_replace(jira ->> 'site', '^[A-Za-z][A-Za-z0-9+.-]*://', ''), '/'))
)
where jira ->> 'site' ~ '^[A-Za-z][A-Za-z0-9+.-]*://|/$';
