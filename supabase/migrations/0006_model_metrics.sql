-- Per-day, per-model-version scorecard: how each deployed model actually performed
-- against what burned. Two purposes, both required by the autonomous retrain loop:
--
--   1. Drift time series — the evidence that decides whether live data has matured
--      enough to be worth retraining on.
--   2. Rollback baseline — risk_scores.model_version attributes every prediction to
--      the model that made it, so a newly promoted model can be compared against its
--      predecessor's trailing performance and automatically reverted if it degrades.
--
-- Recomputing this on demand from risk_scores + feature_history is too slow to do
-- repeatedly (a full-table count already trips Supabase's statement timeout), so it
-- is materialised once per day per version.
--
-- label_def distinguishes the two ground-truth definitions, which are NOT
-- interchangeable: 'fused' (IRWIN+CALFIRE+FIRMS) runs ~2.65x the historical base rate
-- because FIRMS catches industrial/agricultural heat a weather model cannot predict,
-- while 'confirmed' (IRWIN/CALFIRE only) runs ~1.11x and is the fair test. Gates read
-- 'confirmed'; 'fused' is kept for recall monitoring.
--
-- Run this in the Supabase SQL editor (Dashboard -> SQL Editor), as with 0001-0005.
create table if not exists model_metrics (
  id              bigint generated always as identity primary key,
  date            date not null,
  model_version   text not null,
  label_def       text not null check (label_def in ('fused', 'confirmed')),

  n_cells         integer not null,
  n_fires         integer not null,
  base_rate       double precision,

  -- Operational view: what the red and red+yellow tiers actually delivered.
  red_flagged     integer,
  red_hits        integer,
  red_recall      double precision,
  red_precision   double precision,
  red_lift        double precision,
  ry_flagged      integer,
  ry_hits         integer,
  ry_recall       double precision,
  ry_precision    double precision,
  ry_lift         double precision,

  -- Ranking + calibration view: threshold-free, so tier retuning does not
  -- silently break comparability across versions.
  pr_auc          double precision,
  roc_auc         double precision,
  brier           double precision,
  mean_predicted  double precision,

  computed_at     timestamptz not null default now(),
  unique (date, model_version, label_def)
);

create index if not exists model_metrics_lookup_idx
  on model_metrics (model_version, label_def, date desc);

alter table model_metrics enable row level security;
do $$
begin
  create policy "public read model metrics" on model_metrics for select using (true);
exception when duplicate_object then null;
end $$;
