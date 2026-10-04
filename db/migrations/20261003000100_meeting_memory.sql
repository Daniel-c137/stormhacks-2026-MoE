-- Meeting memory: transcript turns and report sections with their embeddings.
-- Private chat is never stored here.
--
-- The vector size is fixed at 768: a migration is static SQL, so it cannot read
-- GEMINI_EMBEDDING_DIM. 768 is the size gemini-embedding-001 produces with
-- output_dimensionality=768 (a quarter of its full 3072 at little loss in retrieval quality,
-- and under pgvector's 2000-dimension index limit). Run with GEMINI_EMBEDDING_DIM=768;
-- changing it needs a new migration and a re-index.
--
-- Search is exact cosine over one team's rows (team_id index), not an approximate HNSW/IVF
-- index: a team's memory is small, and an approximate index combined with the team filter
-- can silently return fewer than k rows on pgvector < 0.8. Add one if a team's memory grows.
--
-- team_id and meeting_id have no foreign keys yet; the teams and meetings tables come with
-- the Postgres store.

create extension if not exists vector;

create table if not exists memory_chunks (
    id text primary key,
    team_id text not null,
    meeting_id text not null,
    kind text not null check (kind in ('transcript', 'summary', 'decision', 'task', 'public_chat')),
    text text not null,
    speaker_id text,
    speaker_name text,
    ref_id text,
    t_start double precision,
    t_end double precision,
    -- the model that produced the embedding; search compares only vectors from one model
    embedding_model text not null,
    embedding vector(768) not null,
    created_at timestamptz not null default now()
);

create index if not exists memory_chunks_team_model_idx on memory_chunks (team_id, embedding_model);
create index if not exists memory_chunks_meeting_kind_idx on memory_chunks (meeting_id, kind);

-- Only the brain's service role (which bypasses RLS) reads and writes memory; no client access.
alter table memory_chunks enable row level security;
