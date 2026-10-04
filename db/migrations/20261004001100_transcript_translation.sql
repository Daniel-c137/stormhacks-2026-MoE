-- Live translation (#106). A meeting translates non-English speech only when its host turned
-- it on before anyone joined. A translated segment's text is the English everyone reads;
-- language is the spoken language's ISO 639 code and original_text the words as said, both
-- null for English.
alter table meetings add column translate boolean not null default false;
alter table transcript_segments add column language text;
alter table transcript_segments add column original_text text;
