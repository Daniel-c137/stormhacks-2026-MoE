-- What was said that led to a decision, oldest first: [{text, t, seg_ids}]. Empty for decisions
-- recorded before chains, which have only their quote.
alter table decisions add column chain jsonb not null default '[]';
