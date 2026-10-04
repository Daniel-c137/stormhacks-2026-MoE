-- Core tables behind the brain's Store (brain/src/brain/pg_store.py).
--
-- Ids are text: the app makes them (UUID strings, or ids like "<meeting>-task-1").
-- Private chat has no table: it is never stored.
-- Every table has row level security on and no policies, so Supabase's anon and authenticated
-- keys read nothing; the brain's own connection owns the tables and bypasses RLS.
-- pgvector and memory_chunks come with 20261003000100_meeting_memory.sql.

create table teams (
    id text primary key,
    name text not null,
    github_repo text,
    jira_project text
);

create table people (
    id text primary key,
    name text not null,
    short text not null,
    initials text not null,
    title text,
    email text,
    photo_url text
);

-- The uploaded profile photo; people.photo_url points at the API route that serves it.
create table person_photos (
    person_id text primary key references people on delete cascade,
    content_type text not null,
    data bytea not null
);

-- person_id has no foreign key: a team may list a member before their profile is saved,
-- and member lists skip ids without a profile.
create table memberships (
    team_id text not null references teams on delete cascade,
    person_id text not null,
    seq bigint generated always as identity,  -- join order
    primary key (team_id, person_id)
);
create index memberships_person_idx on memberships (person_id);

create table team_settings (
    team_id text primary key references teams on delete cascade,
    github jsonb not null,
    jira jsonb not null,
    voice text,
    wake_phrase text,
    sensitivity text not null check (sensitivity in ('quiet', 'balanced', 'eager')),
    interrupt_minutes integer not null,
    who_can_allow text not null check (who_can_allow in ('everyone', 'host'))
);

create table meetings (
    id text primary key,
    seq bigint generated always as identity,  -- creation order, breaks ties in listings
    team_id text not null references teams on delete cascade,
    title text not null,
    status text not null
        check (status in ('scheduled', 'live', 'processing', 'needs_review', 'pushed')),
    code text not null unique,
    host_id text not null,
    participant_ids text[] not null default '{}',
    invitee_ids text[] not null default '{}',
    scheduled_start timestamptz,
    started_at timestamptz,
    ended_at timestamptz,
    duration_min integer,
    jira_keys text[] not null default '{}',
    transcript_deleted_at timestamptz
);
create index meetings_team_idx on meetings (team_id);
create index meetings_retention_idx on meetings (ended_at) where transcript_deleted_at is null;

-- Final segments only. seg_id sorts bytewise so ties order the same everywhere.
create table transcript_segments (
    meeting_id text not null references meetings on delete cascade,
    seg_id text collate "C" not null,
    speaker_id text not null,
    speaker_name text not null,
    text text not null,
    is_final boolean not null,
    t_start double precision not null,
    t_end double precision not null,
    primary key (meeting_id, seg_id)
);
create index transcript_segments_order_idx on transcript_segments (meeting_id, t_start, t_end);

create table public_chat (
    meeting_id text not null references meetings on delete cascade,
    id text not null,
    seq bigint generated always as identity,
    sender_id text not null,
    sender_name text not null,
    is_agent boolean not null,
    text text not null,
    ts timestamptz not null,
    snippet_id text,
    primary key (meeting_id, id)
);

create table agendas (
    meeting_id text primary key references meetings on delete cascade,
    generated_at timestamptz not null,
    updated_at timestamptz
);

create table agenda_items (
    meeting_id text not null references agendas on delete cascade,
    ord integer not null,
    id text not null,
    title text not null,
    why text,
    owner_id text,
    sources jsonb not null default '[]',
    status text not null check (status in ('pending', 'covered', 'skipped')),
    minutes integer,  -- timebox
    added_by text,  -- person id; null when the agent proposed it
    primary key (meeting_id, ord)
);

-- The report's own sections (topics, risks, open questions, ...) are one jsonb document;
-- tasks and decisions are rows so they can be edited, filtered and searched.
create table reports (
    meeting_id text primary key references meetings on delete cascade,
    summary text not null,
    sections jsonb not null default '{}'
);

create table report_progress (
    meeting_id text primary key references meetings on delete cascade,
    steps jsonb not null,
    current_step integer not null,
    done boolean not null,
    error text
);

create table task_drafts (
    id text primary key,
    meeting_id text not null references reports on delete cascade,
    ord integer not null,
    title text not null,
    description text,
    owner_id text,
    due date,
    t double precision,
    quote text,
    include boolean not null default true,
    key text,
    jira_status text not null default 'draft'
        check (jira_status in ('draft', 'todo', 'in_progress', 'done'))
);
create index task_drafts_meeting_idx on task_drafts (meeting_id, ord);
create index task_drafts_owner_idx on task_drafts (owner_id);

-- relation_decision_id has no foreign key: replacing one meeting's report must not touch
-- another meeting's decisions that point at it.
create table decisions (
    id text primary key,
    meeting_id text not null references reports on delete cascade,
    ord integer not null,
    text text not null,
    made_by text not null,
    t double precision not null,
    quote text not null,
    status text not null default 'active' check (status in ('active', 'superseded')),
    relation_type text check (relation_type in ('contradicts', 'superseded_by')),
    relation_decision_id text,
    check ((relation_type is null) = (relation_decision_id is null))
);
create index decisions_meeting_idx on decisions (meeting_id, ord);

alter table teams enable row level security;
alter table people enable row level security;
alter table person_photos enable row level security;
alter table memberships enable row level security;
alter table team_settings enable row level security;
alter table meetings enable row level security;
alter table transcript_segments enable row level security;
alter table public_chat enable row level security;
alter table agendas enable row level security;
alter table agenda_items enable row level security;
alter table reports enable row level security;
alter table report_progress enable row level security;
alter table task_drafts enable row level security;
alter table decisions enable row level security;
