-- Per-date record of whether every ground-truth source answered when that date was
-- labelled. Without this, a labelling run that lost a source is indistinguishable
-- from a genuinely quiet fire day, and the autonomous retrain would silently train
-- on fabricated negatives.
--
-- This is not hypothetical: IRWIN and FIRMS both failed from GitHub Actions starting
-- 2026-07-04 while CAL FIRE kept working. Every run reported success and ~85% of that
-- month's positives were overwritten with zeros. Retraining gates read this table and
-- refuse any date where healthy = false.
--
-- One row per date; the labeller upserts, so this always reflects the current state
-- of that date's labels rather than a history of attempts.
--
-- Run this in the Supabase SQL editor (Dashboard -> SQL Editor), as with 0001-0004.
create table if not exists label_health (
  date              date primary key,
  healthy           boolean not null,
  irwin_ok          boolean not null,
  calfire_ok        boolean not null,
  firms_ok          boolean not null,
  fire_cells        integer,
  confirmed_cells   integer,
  firms_only_cells  integer,
  error             text,
  checked_at        timestamptz not null default now()
);

create index if not exists label_health_healthy_idx on label_health (healthy, date desc);

alter table label_health enable row level security;
do $$
begin
  create policy "public read label health" on label_health for select using (true);
exception when duplicate_object then null;
end $$;
