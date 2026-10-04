-- When the write-up last saved progress, so a write-up that stopped moving (a crashed
-- process) can be retried after a while.
alter table report_progress add column updated_at timestamptz;
