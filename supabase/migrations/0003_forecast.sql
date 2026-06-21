-- Multi-day risk forecast (Route A). One row per cell per target day per run.
-- run_date = when the forecast was computed; target_date = the future day;
-- horizon = days ahead (1..5). The dashboard reads the latest run_date.
create table if not exists forecast_scores (
  id            bigint generated always as identity primary key,
  run_date      date not null,
  target_date   date not null,
  horizon       integer not null,
  grid_id       integer not null references grid_cells (grid_id),
  risk          double precision not null,
  tier          text not null check (tier in ('Red', 'Yellow', 'Green')),
  model_version text,
  unique (run_date, grid_id, horizon)
);
create index if not exists forecast_scores_lookup_idx on forecast_scores (run_date desc, horizon);

alter table forecast_scores enable row level security;
do $$
begin
  create policy "public read forecast" on forecast_scores for select using (true);
exception when duplicate_object then null;
end $$;
