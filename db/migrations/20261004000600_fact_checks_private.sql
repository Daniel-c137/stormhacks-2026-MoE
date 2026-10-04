-- Fact-checks reach only the person who made the claim, as a private chat message from the agent
-- (#113): never the room, so no raised hand and no public visibility. The copy kept for the
-- write-up never records whom it was sent to. finding is what the records show, in one sentence.
alter table fact_checks
    drop column raised_hand,
    drop column visibility,
    drop column recipient_id,
    add column finding text not null default '';

alter table fact_check_state drop column hand_raised_at;
