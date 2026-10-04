-- Fact-checks during a live meeting (#11). Only public checks are stored, for the report; a
-- private check goes to its speaker and is never saved, which the checks below enforce. Times are
-- seconds from the meeting start, like transcript segments.
create table fact_checks (
    meeting_id text not null references meetings on delete cascade,
    id text not null,
    seq bigint generated always as identity,
    claim text not null,
    speaker_name text not null,
    verdict text not null check (verdict in ('supported', 'contradicted', 'unknown')),
    confidence double precision not null check (confidence between 0 and 1),
    severity text not null check (severity in ('low', 'high')),
    snippet_ids jsonb not null default '[]',
    sources jsonb not null default '[]',
    raised_hand boolean not null default false,
    visibility text not null default 'public' check (visibility = 'public'),
    recipient_id text check (recipient_id is null),
    t double precision,
    created_at timestamptz,
    primary key (meeting_id, id)
);

-- Where a live meeting's checks stand, so restarts and replicas behave the same.
create table fact_check_state (
    meeting_id text primary key references meetings on delete cascade,
    checked_until double precision,  -- transcript checked so far; null before any
    checked_at double precision,  -- the latest model check, for the rate limit
    hand_raised_at double precision  -- the latest raised hand, for interrupt_minutes
);

alter table fact_checks enable row level security;
alter table fact_check_state enable row level security;
