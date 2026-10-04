-- The report's summary read aloud (ElevenLabs) for the report page's Listen button. Made once and
-- served until the summary or the voice changes; key hashes the text, voice and model it was made
-- from.
create table report_audio (
    meeting_id text primary key references meetings on delete cascade,
    key text not null,
    content_type text not null,
    data bytea not null
);

alter table report_audio enable row level security;
