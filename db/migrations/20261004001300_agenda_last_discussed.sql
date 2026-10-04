-- When an agenda item was last talked about (seconds from the meeting start), as the tracker
-- labelled it. The tracker covers an item as of this time, and only one that has come up: it is
-- null until then, and again once a person reopens the item.
alter table agenda_items add column last_discussed_t double precision;
