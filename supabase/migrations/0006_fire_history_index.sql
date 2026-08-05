-- Index the fire-recency prior's lookup.
--
-- Scoring now reads confirmed ignitions from feature_history at score time to build
-- the fire-recency features. feature_history is indexed on date but not on has_fire,
-- so selecting confirmed fires over a 120-day window scans roughly half a million
-- rows and intermittently trips the statement timeout. On 2026-08-05 that published a
-- forecast of 4,169 green cells across all six horizons while the same day's nowcast
-- had 143 red — the prior came back empty and the model scored as if nothing had
-- burned all summer.
--
-- A partial index: only the ~0.9% of rows with has_fire = 1 are ever queried this way,
-- so it stays small while turning the scan into a direct lookup.

create index if not exists feature_history_fires_idx
  on feature_history (date, grid_id)
  where has_fire = 1;
