-- Translated speech (#106): text stays the English everyone reads; language is the spoken
-- language's ISO 639-1 code and original_text the words as said, both null for English.
alter table transcript_segments add column language text;
alter table transcript_segments add column original_text text;
